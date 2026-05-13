import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import noteshrink

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class DocumentPreprocessor:
    DEFAULT_CONFIG = {
        # _rescale — upscale if shortest side is below this (0 = disabled)
        "rescale_min_dim": 0,
        # _deskew — correct page rotation
        "deskew": True,
        "deskew_max_angle": 20,
        # _normalize_illumination
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid": (8, 8),
        # _remove_noise — median blur kernel size (0 = disabled, must be odd)
        # disabled by default — median blur erases thin strokes
        "denoise_ksize": 0,
        # _remove_grid — HSV range for light-blue notebook grid lines
        "remove_grid": True,
        "grid_hsv_lo": (85,  20,  80),
        "grid_hsv_hi": (125, 160, 255),
        "grid_min_len_ratio": 0.10,
        "grid_inpaint_radius": 3,
        # _separate_ink (noteshrink)
        "value_threshold": 0.25,
        "sat_threshold": 0.20,
        "num_colors": 8,
        "sample_fraction": 0.05,
        # _erode_dilate — applied to ink mask before binarizing
        #   op: None / 'erode' (thin) / 'dilate' (thicken)
        "morph_op": None,
        "morph_ksize": 2,
        "morph_iters": 1,
        # _add_border — white padding around the final image
        "border_px": 10,
    }

    def __init__(self, config: dict = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

    def preprocess(self, image_path: str) -> np.ndarray:
        img_bgr = self._load_bgr(image_path)
        img_bgr = self._rescale(img_bgr)
        if self.config["deskew"]:
            img_bgr = self._deskew(img_bgr)
        img_bgr = self._normalize_illumination(img_bgr)
        if self.config["denoise_ksize"]:
            img_bgr = self._remove_noise(img_bgr)
        if self.config["remove_grid"]:
            img_bgr = self._remove_grid(img_bgr)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ink_mask = self._separate_ink(img_rgb)
        ink_mask = self._erode_dilate(ink_mask)
        result = self._binarize(ink_mask)
        return self._add_border(result)

    def _load_bgr(self, image_path: str) -> np.ndarray:
        # Read with alpha so we can crop to the opaque bounding box.
        # This removes transparent stripes and shadow margins from book-binding photos
        # before the pipeline runs, preventing those regions from corrupting detection.
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            alpha = img[:, :, 3]
            # Use a high opacity threshold (>200) to exclude semi-transparent shadow
            # gradients at the page edge — these have alpha 20-150 and corrupt detection.
            # A column/row must have >50% of its pixels above the threshold to be kept.
            opaque = (alpha > 200).astype(np.float32)
            rows = opaque.mean(axis=1) > 0.5
            cols = opaque.mean(axis=0) > 0.5
            r0, r1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
            c0, c1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
            img = img[r0:r1, c0:c1]
            # composite cropped region against white
            a = img[:, :, 3:4].astype(np.float32) / 255.0
            bgr = img[:, :, :3].astype(np.float32)
            img = (bgr * a + np.full_like(bgr, 255.0) * (1.0 - a)).astype(np.uint8)
        return img

    def _rescale(self, img_bgr: np.ndarray) -> np.ndarray:
        min_dim = self.config["rescale_min_dim"]
        if not min_dim:
            return img_bgr
        h, w = img_bgr.shape[:2]
        short = min(h, w)
        if short >= min_dim:
            return img_bgr
        scale = min_dim / short
        return cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)

    def _deskew(self, img_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # Downscale for speed; deskew doesn't need full resolution.
        small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        thresh = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Projection-profile sweep: the correct rotation angle maximises the
        # variance of row sums (text rows become sharp dark bands when horizontal).
        max_angle = self.config["deskew_max_angle"]
        h, w = thresh.shape
        center = (w / 2, h / 2)
        best_angle, best_score = 0.0, -1.0
        for angle in np.arange(-max_angle, max_angle + 0.5, 0.5):
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_NEAREST)
            score = float(rotated.sum(axis=1).var())
            if score > best_score:
                best_score, best_angle = score, angle

        if abs(best_angle) < 0.5:
            return img_bgr
        h, w = img_bgr.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), best_angle, 1.0)
        return cv2.warpAffine(img_bgr, M, (w, h),
                              flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def _normalize_illumination(self, img: np.ndarray) -> np.ndarray:
        # CLAHE on the L channel corrects uneven lighting (shadows from book binding,
        # phone-camera gradients) locally so noteshrink sees a uniformly bright paper.
        # Global methods would brighten shadows but also blow out already-bright regions.
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.config["clahe_clip_limit"],
            tileGridSize=self.config["clahe_tile_grid"],
        )
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def _remove_noise(self, img_bgr: np.ndarray) -> np.ndarray:
        k = self.config["denoise_ksize"]
        if k % 2 == 0:
            k += 1
        return cv2.medianBlur(img_bgr, k)

    def _remove_grid(self, img_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lo = np.array(self.config["grid_hsv_lo"], dtype=np.uint8)
        hi = np.array(self.config["grid_hsv_hi"], dtype=np.uint8)
        color_mask = cv2.inRange(hsv, lo, hi)

        h, w = color_mask.shape
        min_len = max(15, int(min(h, w) * self.config["grid_min_len_ratio"]))
        h_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
        v_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len))
        grid_mask = cv2.bitwise_or(
            cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, h_kern),
            cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, v_kern),
        )

        if grid_mask.max() == 0:
            return img_bgr

        # Widen mask by 1 px on each side to cover full line width after opening
        grid_mask = cv2.dilate(grid_mask, np.ones((3, 3), np.uint8))
        return cv2.inpaint(img_bgr, grid_mask, self.config["grid_inpaint_radius"],
                           cv2.INPAINT_TELEA)

    def _separate_ink(self, img_rgb: np.ndarray) -> np.ndarray:
        opts = SimpleNamespace(
            value_threshold=self.config["value_threshold"],
            sat_threshold=self.config["sat_threshold"],
            num_colors=self.config["num_colors"],
            sample_fraction=self.config["sample_fraction"],
            quiet=True,
        )
        samples = noteshrink.sample_pixels(img_rgb, opts)
        palette = noteshrink.get_palette(samples, opts)
        labels = noteshrink.apply_palette(img_rgb, palette, opts)
        return (labels != 0).astype(np.uint8) * 255

    def _erode_dilate(self, ink_mask: np.ndarray) -> np.ndarray:
        op = self.config["morph_op"]
        if not op:
            return ink_mask
        k = self.config["morph_ksize"]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        fn = cv2.dilate if op == "dilate" else cv2.erode
        return fn(ink_mask, kernel, iterations=self.config["morph_iters"])

    def _binarize(self, ink_mask: np.ndarray) -> np.ndarray:
        # Invert to black-on-white; convert to 3-channel BGR for YOLO compatibility.
        return cv2.cvtColor(cv2.bitwise_not(ink_mask), cv2.COLOR_GRAY2BGR)

    def _add_border(self, img_bgr: np.ndarray) -> np.ndarray:
        px = self.config["border_px"]
        if not px:
            return img_bgr
        return cv2.copyMakeBorder(img_bgr, px, px, px, px,
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))


def _process_file(src: Path, preprocessed_dir: Path, compare_dir: Path,
                  preprocessor: DocumentPreprocessor) -> None:
    original = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if original is None:
        print(f"Cannot read: {src}")
        return
    if original.ndim == 3 and original.shape[2] == 4:
        a = original[:, :, 3:4].astype(np.float32) / 255.0
        original = (original[:, :, :3].astype(np.float32) * a +
                    np.full_like(original[:, :, :3], 255, dtype=np.float32) * (1 - a)).astype(np.uint8)

    result = preprocessor.preprocess(str(src))

    h = min(original.shape[0], 900)
    scale = h / original.shape[0]
    orig_disp = cv2.resize(original, (int(original.shape[1] * scale), h))
    res_disp  = cv2.resize(result,   (int(result.shape[1]   * scale), h))
    combined  = np.hstack([orig_disp, res_disp])

    out_path     = str(preprocessed_dir / f"{src.stem}.jpg")
    compare_path = str(compare_dir      / f"{src.stem}_compare.jpg")
    cv2.imwrite(out_path, result)
    cv2.imwrite(compare_path, combined)
    print(f"  {src.name} -> {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <image_path_or_folder>")
        sys.exit(1)

    target = Path(sys.argv[1])
    preprocessor = DocumentPreprocessor()

    if target.is_dir():
        images = sorted(p for p in target.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not images:
            print(f"No images found in {target}")
            sys.exit(1)
        preprocessed_dir = target.parent / f"{target.name}_preprocessed"
        compare_dir      = target.parent / f"{target.name}_compare"
        preprocessed_dir.mkdir(exist_ok=True)
        compare_dir.mkdir(exist_ok=True)
        print(f"Processing {len(images)} images from {target}/")
        for img_path in images:
            _process_file(img_path, preprocessed_dir, compare_dir, preprocessor)
        print(f"Done. Results in {preprocessed_dir}/")
    else:
        preprocessed_dir = target.parent / "preprocessed"
        compare_dir      = target.parent / "compare"
        preprocessed_dir.mkdir(exist_ok=True)
        compare_dir.mkdir(exist_ok=True)
        _process_file(target, preprocessed_dir, compare_dir, preprocessor)
        subprocess.run(["open", str(compare_dir / f"{target.stem}_compare.jpg")], check=True)

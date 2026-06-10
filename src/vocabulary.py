"""Vocabulary loader for Polish medical text generation.

Loads resource files (drug names, ICD-10 codes, abbreviations, dosages,
diagnoses, patient names) and provides random sampling functions.
"""

import os
import random
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import openpyxl


# ICD-10 code pattern: letter + 2 digits, optionally dot + 1-2 digits
_ICD_CODE_PATTERN_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.")

CATEGORIES = [
    "drug",
    "icd_code",
    "icd_description",
    "icd_full",
    "abbreviation",
    "dosage",
    "diagnosis",
    "patient_name",
    "date",
    "pesel",
    "address",
    "phone",
    "doctor_name",
    "hospital_name",
    "department",
    "age",
    "year",
    "city",
    "day_and_month",
    "last_2_digits_year",
    "approval",
]


# Possible texts for "przyczyna ewentualnej odmowy/zalecenia" field
_APPROVAL_TEXTS = [
    "Skierowanie ważne 30 dni",
    "Pilne badanie",
    "Zalecana hospitalizacja",
    "Wymagane konsultacje przed przyjęciem",
    "Termin do uzgodnienia",
    "Kontrola po 4 tygodniach",
    "Brak miejsc w oddziale",
    "Pacjent nie zgłosił się w terminie",
    "Skierowanie do realizacji w trybie planowym",
    "Zalecane wykonanie badań laboratoryjnych",
    "Wymagane EKG przed przyjęciem",
    "Pacjent w stanie stabilnym",
    "Zalecana konsultacja kardiologiczna",
    "Hospitalizacja w trybie pilnym",
    "Kontynuacja leczenia szpitalnego",
    "Skierowanie zaakceptowane",
    "Termin przyjęcia do uzgodnienia telefonicznego",
    "Wymagana konsultacja anestezjologiczna",
    "Pacjent zgłasza się z opiekunem",
    "Diagnostyka w warunkach szpitalnych",
]


# Procedural data sources for form-filling categories
_STREETS = [
    "Marszałkowska", "Piotrkowska", "Krakowska", "Warszawska", "Słowackiego",
    "Mickiewicza", "Sienkiewicza", "Reymonta", "Kościuszki", "Piłsudskiego",
    "3 Maja", "11 Listopada", "Niepodległości", "Wolności", "Pokoju",
    "Zielona", "Krótka", "Długa", "Polna", "Leśna", "Łąkowa", "Słoneczna",
    "Akacjowa", "Lipowa", "Brzozowa", "Sosnowa", "Kwiatowa", "Ogrodowa",
    "Szkolna", "Kościelna", "Rynkowa", "Główna", "Boczna", "Nowa",
    "Jana Pawła II", "Stefana Batorego", "Bolesława Chrobrego", "Władysława Jagiełły",
]

_CITIES = [
    "Warszawa", "Kraków", "Łódź", "Wrocław", "Poznań", "Gdańsk", "Szczecin",
    "Bydgoszcz", "Lublin", "Białystok", "Katowice", "Gdynia", "Częstochowa",
    "Radom", "Sosnowiec", "Toruń", "Kielce", "Rzeszów", "Gliwice", "Zabrze",
    "Olsztyn", "Bielsko-Biała", "Bytom", "Zielona Góra", "Rybnik", "Ruda Śląska",
    "Tychy", "Opole", "Gorzów Wielkopolski", "Dąbrowa Górnicza", "Płock",
    "Elbląg", "Wałbrzych", "Włocławek", "Tarnów", "Chorzów", "Koszalin",
    "Kalisz", "Legnica", "Grudziądz", "Słupsk", "Jaworzno", "Jastrzębie-Zdrój",
    "Skierniewice",
]

_HOSPITAL_TEMPLATES = [
    "Szpital Wojewódzki im. {patron} w {city}",
    "Szpital Miejski w {city}",
    "Szpital Specjalistyczny w {city}",
    "Wojewódzki Szpital Zespolony w {city}",
    "Szpital Kliniczny im. {patron} w {city}",
    "Centrum Medyczne {patron} w {city}",
    "SP ZOZ w {city}",
    "Szpital Powiatowy w {city}",
]

_HOSPITAL_PATRONS = [
    "Jana Pawła II", "Mikołaja Kopernika", "św. Łukasza", "św. Anny",
    "M. Kopernika", "L. Rydygiera", "T. Marcinkowskiego",
    "ks. A. Markwarta", "T. Chałubińskiego", "J. Korczaka",
]

_DEPARTMENTS = [
    "Kardiologii", "Neurologii", "Ortopedii", "Chirurgii Ogólnej",
    "Chirurgii Naczyniowej", "Onkologii", "Pulmonologii", "Gastroenterologii",
    "Endokrynologii", "Reumatologii", "Nefrologii", "Urologii",
    "Ginekologii i Położnictwa", "Pediatrii", "Neonatologii",
    "Anestezjologii i Intensywnej Terapii", "Rehabilitacji",
    "Psychiatrii", "Dermatologii", "Okulistyki", "Otolaryngologii",
    "Chorób Wewnętrznych", "Chorób Zakaźnych", "Hematologii",
    "Chirurgii Urazowo-Ortopedycznej", "Neurochirurgii",
]

_DOCTOR_TITLES = ["dr", "lek.", "lek. med.", "dr n. med.", "dr hab.", "prof. dr hab."]


class Vocabulary:
    """Holds all loaded vocabulary data and provides random sampling."""

    def __init__(self, resource_dir: str = "resources") -> None:
        """Load all resource files from the given directory.

        Args:
            resource_dir: Path to the resources directory.
        """
        self.resource_dir = Path(resource_dir)
        self.drugs: list[str] = []
        self.icd_entries: list[tuple[str, str]] = []  # (code, description)
        self.abbreviations: list[str] = []
        self.dosages: list[str] = []
        self.diagnoses: list[str] = []
        self.first_names_male: list[str] = []
        self.first_names_female: list[str] = []
        self.last_names: list[str] = []

        self._load_all()

    def _load_all(self) -> None:
        """Load all resource files."""
        self.drugs = self._load_drugs()
        self.icd_entries = self._load_icd10()
        self.abbreviations = self._load_txt("medical_abbreviations.txt")
        self.dosages = self._load_txt("dosage_patterns.txt")
        self.diagnoses = self._load_txt("common_diagnoses.txt")
        self.first_names_male = self._load_txt("first_names_male.txt")
        self.first_names_female = self._load_txt("first_names_female.txt")
        self.last_names = self._load_txt("last_names.txt")

    def _load_txt(self, filename: str) -> list[str]:
        """Load a text file, skipping comments (#) and blank lines.

        Args:
            filename: Name of the file inside the resource directory.

        Returns:
            List of non-empty, non-comment lines.
        """
        path = self.resource_dir / filename
        if not path.exists():
            return []
        lines = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    lines.append(stripped)
        return lines

    def _load_drugs(self) -> list[str]:
        """Load drug names from the Excel file (column 2: Nazwa Produktu Leczniczego).

        Returns:
            List of drug name strings.
        """
        path = self.resource_dir / "drug_names.xlsx"
        if not path.exists():
            return []
        try:
            wb = openpyxl.load_workbook(path, read_only=True)
            ws = wb.active
            drugs = []
            for row in ws.iter_rows(min_row=2, min_col=2, max_col=2, values_only=True):
                val = row[0]
                if val and str(val).strip():
                    drugs.append(str(val).strip())
            wb.close()
            return drugs
        except Exception:
            return []

    def _load_icd10(self) -> list[tuple[str, str]]:
        """Parse ICD-10 XML and extract (code, Polish description) pairs.

        Returns:
            List of (code, description) tuples for leaf-level codes.
        """
        path = self.resource_dir / "icd-10.xml"
        if not path.exists():
            return []
        try:
            tree = ET.parse(path)
            root = tree.getroot()
        except ET.ParseError:
            return []

        ns = {"h": "http://rsk.rejestrymedyczne.csioz.gov.pl"}
        entries: list[tuple[str, str]] = []

        for node in root.iter(f"{{{ns['h']}}}node"):
            code = node.get("code", "").strip()
            if not code:
                continue
            # Only keep actual ICD-10 codes (e.g. A00, A00.0), skip ranges like A00-A09
            if "–" in code or "-" in code:
                continue
            if len(code) < 3:
                continue
            name_el = node.find("h:name", ns)
            if name_el is not None and name_el.text and name_el.text.strip():
                entries.append((code, name_el.text.strip()))

        return entries

    @staticmethod
    def _feminize_last_name(last_name: str) -> str:
        """Convert a male Polish last name to its female form.

        Args:
            last_name: Male form of the last name.

        Returns:
            Female form of the last name.
        """
        if last_name.endswith("dzki"):
            return last_name[:-4] + "dzka"
        if last_name.endswith("cki"):
            return last_name[:-3] + "cka"
        if last_name.endswith("ski"):
            return last_name[:-3] + "ska"
        return last_name

    @staticmethod
    def _generate_pesel(birth_date: Optional[date] = None, female: Optional[bool] = None) -> str:
        """Generate a valid Polish PESEL number.

        PESEL structure: YYMMDDSSSSQ where:
        - YYMMDD = birth date (month offset by 20 for 2000s)
        - SSSS = serial number (odd last digit = male, even = female)
        - Q = checksum digit

        Args:
            birth_date: Date of birth. Random if None.
            female: Whether the person is female. Random if None.

        Returns:
            11-digit PESEL string with valid checksum.
        """
        if birth_date is None:
            start = date(1950, 1, 1)
            end = date(2005, 12, 31)
            birth_date = start + timedelta(days=random.randint(0, (end - start).days))

        if female is None:
            female = random.choice([True, False])

        year = birth_date.year
        month = birth_date.month
        day = birth_date.day

        yy = year % 100
        if 2000 <= year <= 2099:
            mm = month + 20
        elif 1900 <= year <= 1999:
            mm = month
        else:
            mm = month

        # Serial number: 3 random digits + sex digit
        serial_3 = random.randint(0, 999)
        if female:
            sex_digit = random.choice([0, 2, 4, 6, 8])
        else:
            sex_digit = random.choice([1, 3, 5, 7, 9])

        digits = list(f"{yy:02d}{mm:02d}{day:02d}{serial_3:03d}{sex_digit}")
        digits = [int(d) for d in digits]

        # Checksum
        weights = [1, 3, 7, 9, 1, 3, 7, 9, 1, 3]
        total = sum(d * w for d, w in zip(digits, weights))
        checksum = (10 - (total % 10)) % 10

        digits.append(checksum)
        return "".join(str(d) for d in digits)

    @staticmethod
    def _random_date() -> str:
        """Generate a random date in DD.MM.YYYY format (2015-2025).

        Returns:
            Date string in DD.MM.YYYY format.
        """
        start = date(2015, 1, 1)
        end = date(2025, 12, 31)
        d = start + timedelta(days=random.randint(0, (end - start).days))
        return d.strftime("%d.%m.%Y")

    def get_random_text(self, category: Optional[str] = None) -> tuple[str, str]:
        """Return a random text sample from the given category.

        Args:
            category: One of the CATEGORIES, or None for random.

        Returns:
            Tuple of (text, category_name).
        """
        if category is None:
            category = random.choice(CATEGORIES)

        if category == "drug":
            if not self.drugs:
                return ("Amoksycylina", "drug")
            return (random.choice(self.drugs), "drug")

        elif category == "icd_code":
            if not self.icd_entries:
                return ("J06.9", "icd_code")
            code, _ = random.choice(self.icd_entries)
            return (code, "icd_code")

        elif category == "icd_description":
            if not self.icd_entries:
                return ("Ostre zakażenie górnych dróg oddechowych", "icd_description")
            _, desc = random.choice(self.icd_entries)
            return (desc, "icd_description")

        elif category == "icd_full":
            if not self.icd_entries:
                return ("J06.9 Ostre zakażenie górnych dróg oddechowych", "icd_full")
            code, desc = random.choice(self.icd_entries)
            return (f"{code} {desc}", "icd_full")

        elif category == "abbreviation":
            if not self.abbreviations:
                return ("Rp.", "abbreviation")
            return (random.choice(self.abbreviations), "abbreviation")

        elif category == "dosage":
            if not self.dosages:
                return ("1-0-1", "dosage")
            return (random.choice(self.dosages), "dosage")

        elif category == "diagnosis":
            if not self.diagnoses:
                return ("nadciśnienie tętnicze", "diagnosis")
            return (random.choice(self.diagnoses), "diagnosis")

        elif category == "patient_name":
            is_female = random.choice([True, False])
            if is_female:
                first = random.choice(self.first_names_female) if self.first_names_female else "Anna"
                last = random.choice(self.last_names) if self.last_names else "Kowalska"
                last = self._feminize_last_name(last)
            else:
                first = random.choice(self.first_names_male) if self.first_names_male else "Jan"
                last = random.choice(self.last_names) if self.last_names else "Kowalski"
            return (f"{first} {last}", "patient_name")

        elif category == "date":
            return (self._random_date(), "date")

        elif category == "pesel":
            return (self._generate_pesel(), "pesel")

        elif category == "address":
            return (self._random_address(), "address")

        elif category == "phone":
            return (self._random_phone(), "phone")

        elif category == "doctor_name":
            return (self._random_doctor_name(), "doctor_name")

        elif category == "hospital_name":
            return (self._random_hospital_name(), "hospital_name")

        elif category == "department":
            return (self._random_department(), "department")

        elif category == "age":
            return (str(random.randint(1, 99)), "age")

        elif category == "year":
            return (str(random.randint(2020, 2025)), "year")

        elif category == "city":
            return (random.choice(_CITIES), "city")

        elif category == "day_and_month":
            d = self._random_day_and_month()
            return (d, "day_and_month")

        elif category == "last_2_digits_year":
            return (f"{random.randint(20, 25):02d}", "last_2_digits_year")

        elif category == "approval":
            return (random.choice(_APPROVAL_TEXTS), "approval")

        elif category == "text":
            # Generic alphabetic content: random pick from any text-like category
            text_categories = [
                "patient_name", "patient_name",  # name appears more often
                "diagnosis", "icd_description",
                "address", "city",
                "hospital_name", "doctor_name", "department",
                "approval", "drug",
            ]
            sub = random.choice(text_categories)
            content, _ = self.get_random_text(sub)
            return (content, "text")

        elif category == "number":
            # Generic numeric content: random pick from any number-like category
            number_categories = [
                "pesel", "date", "date",  # date appears more often
                "phone", "icd_code",
                "year", "age",
                "day_and_month", "last_2_digits_year",
            ]
            sub = random.choice(number_categories)
            content, _ = self.get_random_text(sub)
            return (content, "number")

        else:
            return self.get_random_text(None)

    @staticmethod
    def _random_day_and_month() -> str:
        """Generate a random DD.MM string."""
        start = date(2020, 1, 1)
        end = date(2025, 12, 31)
        d = start + timedelta(days=random.randint(0, (end - start).days))
        return d.strftime("%d.%m")

    @staticmethod
    def _random_address() -> str:
        """Generate a random Polish address: street + number, postal code + city."""
        street = random.choice(_STREETS)
        number = random.randint(1, 250)
        if random.random() < 0.3:
            number = f"{number}/{random.randint(1, 50)}"
        postal = f"{random.randint(10, 99)}-{random.randint(100, 999)}"
        city = random.choice(_CITIES)
        if random.random() < 0.5:
            return f"ul. {street} {number}, {postal} {city}"
        return f"{street} {number}, {city}"

    @staticmethod
    def _random_phone() -> str:
        """Generate a random Polish phone number."""
        formats = [
            lambda: f"+48 {random.randint(500, 799)} {random.randint(100, 999)} {random.randint(100, 999)}",
            lambda: f"{random.randint(500, 799)}-{random.randint(100, 999)}-{random.randint(100, 999)}",
            lambda: f"{random.randint(500, 799)} {random.randint(100, 999)} {random.randint(100, 999)}",
            lambda: f"({random.randint(12, 99)}) {random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(10, 99)}",
        ]
        return random.choice(formats)()

    def _random_doctor_name(self) -> str:
        """Generate a random doctor name with title."""
        title = random.choice(_DOCTOR_TITLES)
        is_female = random.choice([True, False])
        if is_female:
            first = random.choice(self.first_names_female) if self.first_names_female else "Anna"
            last = random.choice(self.last_names) if self.last_names else "Kowalska"
            last = self._feminize_last_name(last)
        else:
            first = random.choice(self.first_names_male) if self.first_names_male else "Jan"
            last = random.choice(self.last_names) if self.last_names else "Kowalski"
        return f"{title} {first} {last}"

    @staticmethod
    def _random_hospital_name() -> str:
        """Generate a random Polish hospital name."""
        template = random.choice(_HOSPITAL_TEMPLATES)
        return template.format(
            patron=random.choice(_HOSPITAL_PATRONS),
            city=random.choice(_CITIES),
        )

    @staticmethod
    def _random_department() -> str:
        """Generate a random hospital department name."""
        return f"Oddział {random.choice(_DEPARTMENTS)}"

    def get_random_phrase(self) -> tuple[str, str]:
        """Generate a realistic short medical phrase combining elements.

        Returns:
            Tuple of (phrase_text, category_name).
        """
        templates = [
            self._phrase_prescription,
            self._phrase_dosage_instruction,
            self._phrase_patient_label,
            self._phrase_pesel_label,
            self._phrase_diagnosis_label,
            self._phrase_diagnosis_plain,
            self._phrase_date_label,
            self._phrase_drug_with_dose,
        ]
        func = random.choice(templates)
        return func()

    def _phrase_prescription(self) -> tuple[str, str]:
        drug, _ = self.get_random_text("drug")
        dose = random.choice(["100mg", "250mg", "500mg", "1000mg", "50mg", "200mg", "75mg"])
        return (f"Rp. {drug} {dose}", "drug")

    def _phrase_dosage_instruction(self) -> tuple[str, str]:
        dosage, _ = self.get_random_text("dosage")
        duration = random.choice(["przez 5 dni", "przez 7 dni", "przez 10 dni", "przez 14 dni", "przez 3 dni"])
        return (f"{dosage} {duration}", "dosage")

    def _phrase_patient_label(self) -> tuple[str, str]:
        name, _ = self.get_random_text("patient_name")
        return (f"Pacjent: {name}", "patient_name")

    def _phrase_pesel_label(self) -> tuple[str, str]:
        pesel, _ = self.get_random_text("pesel")
        return (f"PESEL: {pesel}", "pesel")

    def _phrase_diagnosis_label(self) -> tuple[str, str]:
        code, _ = self.get_random_text("icd_code")
        return (f"Rozpoznanie: {code}", "icd_code")

    def _phrase_diagnosis_plain(self) -> tuple[str, str]:
        return self.get_random_text("diagnosis")

    def _phrase_date_label(self) -> tuple[str, str]:
        d, _ = self.get_random_text("date")
        prefix = random.choice(["Data:", "Data wizyty:", "Data badania:", ""])
        if prefix:
            return (f"{prefix} {d}", "date")
        return (d, "date")

    def _phrase_drug_with_dose(self) -> tuple[str, str]:
        drug, _ = self.get_random_text("drug")
        dose = random.choice(["100mg", "250mg", "500mg", "1g", "50mg", "200mg"])
        freq = random.choice(["1x dz.", "2x dz.", "3x dz.", "co 8h", "co 12h"])
        return (f"{drug} {dose} {freq}", "drug")

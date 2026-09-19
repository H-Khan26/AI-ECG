# ai_ecg/selection.py

import os
import glob
import shutil
import xml.etree.ElementTree as ET

from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# XPaths inside each XML for patient ID and acquisition date
XML_ID_XPATH   = "./PatientDemographics/IP_PATIENT_ID"
XML_DATE_XPATH = "./TestDemographics/AcquisitionDate"

# List of folders to search for XMLs
DATA_FOLDERS = [
    "EKG_DATA_CTRL",
    "EKG_DATA_STUDY",
]

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _parse_date(s: str) -> datetime.date:
    """
    Parse a date string either in "MM/DD/YYYY" or "DD-MM-YYYY" form.
    """
    s = s.strip()
    if "/" in s:
        return datetime.strptime(s, "%m/%d/%Y").date()
    elif "-" in s:
        return datetime.strptime(s, "%d-%m-%Y").date()
    else:
        raise ValueError(f"Unrecognized date format: {s!r}")

def _get_xml_info(xml_path: str) -> tuple[str, datetime.date]:
    """
    From one ECG XML, extract (ippat, acquisition_date).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # patient ID
    pid_node = root.find(XML_ID_XPATH)
    ippat = pid_node.text.strip() if (pid_node is not None and pid_node.text) else None

    # acquisition date
    date_node = root.find(XML_DATE_XPATH)
    if date_node is None or not date_node.text:
        acq_date = None
    else:
        acq_date = _parse_date(date_node.text)

    return ippat, acq_date

# ─── MAIN SELECTION FUNCTION ──────────────────────────────────────────────────

def select_cases(
    cases_csv: str,
    output_dir: str,
    id_col: str = "IP_PATIENT_ID",
    date_col: str = "ICI_START_DATE",
    days_before: int = 1460,
    days_after: int = 30,
    data_folders: list[str] = None,
):
    """
    Copy all ECG XML files for patients listed in `cases_csv` whose
    acquisition date is within [days_before before, days_after after]
    the ICI start date.

    Args:
      cases_csv:      Path to CSV with columns [id_col, date_col].
      output_dir:     Directory where matched XMLs will be copied.
                      Creates subfolder named after the CSV basename.
      id_col:         Column name in cases_csv for the IP_PATIENT_ID.
      date_col:       Column name in cases_csv for the ICI start date.
      days_before:    Max days *before* ICI to include.
      days_after:     Max days *after* ICI to include.
      data_folders:   List of folders to search for XML files.
                      Defaults to module‐level DATA_FOLDERS.
    """
    data_folders = data_folders or DATA_FOLDERS

    # 1) load cases CSV
    cases = pd.read_csv(cases_csv, dtype=str)
    cases[id_col] = cases[id_col].str.strip()
    cases[date_col] = pd.to_datetime(
        cases[date_col].str.strip(),
        infer_datetime_format=True,
        dayfirst=False
    ).dt.date
    cases = cases.set_index(id_col)[date_col]

    # 2) prepare output
    csv_tag = Path(cases_csv).stem
    dest_base = Path(output_dir) / csv_tag
    dest_base.mkdir(parents=True, exist_ok=True)

    # 3) iterate XMLs and copy matches
    total_checked = 0
    total_copied = 0
    patients_matched = set()

    for folder in data_folders:
        pattern = os.path.join(folder, "**", "*.xml")
        for xml_path in glob.glob(pattern, recursive=True):
            total_checked += 1
            try:
                ippat, acq_date = _get_xml_info(xml_path)
            except Exception:
                continue

            if ippat not in cases.index or acq_date is None:
                continue

            ici_date = cases[ippat]
            delta = (acq_date - ici_date).days

            if -days_before <= delta <= days_after:
                # copy to dest
                dest_path = dest_base / os.path.basename(xml_path)
                shutil.copy2(xml_path, dest_path)
                total_copied += 1
                patients_matched.add(ippat)

    print(f"Checked {total_checked} XMLs; copied {total_copied} to '{dest_base}'")
    print(f"Unique patients matched: {len(patients_matched)} out of {len(cases)}")

    return {
        "checked": total_checked,
        "copied": total_copied,
        "patients_matched": patients_matched,
    }


# ─── COMMAND‐LINE ENTRYPOINT ─────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Select ECG XMLs by ICI cases and date window")
    p.add_argument("cases_csv",   help="CSV of cases with IP_PATIENT_ID and ICI_START_DATE")
    p.add_argument("output_dir",  help="Where to copy matched XMLs")
    p.add_argument("--before",    type=int, default=1460, help="Days before ICI")
    p.add_argument("--after",     type=int, default=30,   help="Days after ICI")
    p.add_argument(
        "--folders",
        nargs="+",
        default=None,
        help="List of ECG data folders (defaults to module setting)"
    )
    args = p.parse_args()

    select_cases(
        cases_csv=args.cases_csv,
        output_dir=args.output_dir,
        days_before=args.before,
        days_after=args.after,
        data_folders=args.folders
    )

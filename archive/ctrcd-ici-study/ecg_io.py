# ai_ecg/ecg_io.py

import os
import glob
import json
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from scipy.signal import decimate


# ─── CONFIG ───────────────────────────────────────────────────────────────────

# Folders containing your XML data (edit these to your local paths)
DATA_FOLDERS = [
    "EKG_DATA_CTRL",
    "EKG_DATA_STUDY",
]

# Where to write the master CSV
OUTPUT_CSV = "data/ecg_master.csv"


# ─── METADATA & INTERVAL FIELDS ───────────────────────────────────────────────

METADATA_FIELDS = {
    "IP_PATIENT_ID":   "./PatientDemographics/IP_PATIENT_ID",
    "PatientAge":      "./PatientDemographics/PatientAge",
    "Gender":          "./PatientDemographics/Gender",
    "DateOfBirth":     "./PatientDemographics/DateofBirth",
    "Race":            "./PatientDemographics/Race",
    "SamplingRate_Hz": "./RestingECGMeasurements/ECGSampLeBase",
    "NumLeads":        "./Waveform/NumberofLeads",
    "AtrialRate":      "./RestingECGMeasurements/AtrialRate",
    "VentricularRate": "./RestingECGMeasurements/VentricularRate",
}

INTERVAL_FIELDS = {
    "HeartRate":   "./RestingECGMeasurements/VentricularRate",
    "QTInterval":  "./RestingECGMeasurements/QTInterval",
    "QRSDuration": "./RestingECGMeasurements/QRSDuration",
    "TAxis":       "./RestingECGMeasurements/TAxis",
}

ABNORMAL_KEYWORDS = [
    "1avb", "atrial fibrillation", "rbbb", "lbbb",
    "st-t change", "lvh", "pac", "pvc", "low voltage",
    "paced rhythm",
]


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def hex64_to_numerical(hex64_str):
    """
    Decodes a Base64-encoded hex string and converts it into numerical values.

    Args:
        hex64_str (str): Base64-encoded string of hexadecimal values.

    Returns:
        list: A list of numerical values (integers).
    """
    # Decode the Base64 string
    decoded_bytes = base64.b64decode(hex64_str)

    # Convert bytes to numerical values
    numerical_values = list(decoded_bytes)

    return numerical_values

def hex64_to_signed_numerical(hex64_str, expected_byte_count, expected_samples, lead_high_limit, lead_low_limit):
    """
    Decodes a Base64-encoded string containing 16-bit signed integer values,
    ensuring it matches the expected byte count and produces the correct number of samples.
    Also, checks that values fall within the provided lead high and low limits.

    Args:
        hex64_str (str): Base64-encoded string of 16-bit integer values.
        expected_samples (int): Expected number of 16-bit signed integer values.
        expected_byte_count (int): Expected number of bytes in the decoded data.
        lead_high_limit (int): Upper bound for valid numerical values.
        lead_low_limit (int): Lower bound for valid numerical values.

    Returns:
        list: A list of numerical values (signed 16-bit integers) within limits.
    """
    # Decode Base64 string to raw bytes
    decoded_bytes = base64.b64decode(hex64_str)

    # Validate byte count
    actual_byte_count = len(decoded_bytes)

    if actual_byte_count != expected_byte_count:
        raise ValueError(
            f"Byte count mismatch: Expected {expected_byte_count} bytes, but got {actual_byte_count} bytes."
        )

    # Ensure the correct number of samples
    expected_bytes_from_samples = expected_samples * 2  # Each 16-bit sample = 2 bytes

    if actual_byte_count != expected_bytes_from_samples:
        raise ValueError(
            f"Sample count mismatch: Expected {expected_samples} samples ({expected_bytes_from_samples} bytes), "
            f"but got {actual_byte_count} bytes."
        )

    # Convert bytes to 16-bit signed integers
    numerical_values = list(struct.unpack(f'{expected_samples}h', decoded_bytes))

    # Apply lead high & low limit constraints (clamping)
    clamped_values = [
        max(lead_low_limit, min(num, lead_high_limit)) for num in numerical_values
    ]

    return clamped_values

def normalize_to_250hz(sig: list[float]) -> list[float]:
    """
    Given a raw list of samples (any length), return exactly 2500 points:
     - if len(sig)==5000, decimate by 2
     - if len(sig)==2500, leave unchanged
     - otherwise linearly interpolate to 2500
    """
    arr = np.array(sig, dtype=float)
    n = arr.shape[0]

    if n == 5000:
        return decimate(arr, 2, ftype="fir").tolist()
    if n == 2500:
        return arr.tolist()

    # fallback interpolation
    return np.interp(
        np.linspace(0, n, 2500, endpoint=False),
        np.arange(n),
        arr
    ).tolist()


def decode_lead(ld: ET.Element) -> tuple[str, list[float]]:
    """
    Parse one <LeadData> element, decode its Base64 into signed ints,
    and return (lead_id, [raw_signal]).
    """
    lead_id      = ld.findtext("LeadID", "").strip()
    byte_count   = int(ld.findtext("LeadByteCountTotal", "0") or 0)
    sample_count = int(ld.findtext("LeadSampleCountTotal", "0") or 0)
    high_limit   = float(ld.findtext("LeadHighLimit", "0") or 0)
    low_limit    = float(ld.findtext("LeadLowLimit", "0") or 0)
    b64_data     = ld.findtext("WaveFormData", "").strip()

    try:
        sig = hex64_to_signed_numerical(
            b64_data,
            expected_byte_count=byte_count,
            expected_samples=sample_count,
            lead_high_limit=high_limit,
            lead_low_limit=low_limit
        )
    except Exception:
        sig = []  # on error, return empty list

    return lead_id, sig


def is_abnormal_ecg(root: ET.Element) -> bool:
    """
    Scan all DiagnosisStatement text for any abnormal keywords.
    """
    for stmt in root.findall(".//Diagnosis/DiagnosisStatement/StmtText"):
        text = (stmt.text or "").lower()
        for kw in ABNORMAL_KEYWORDS:
            if kw in text:
                return True
    return False


# ─── MAIN XML→CSV FUNCTION ────────────────────────────────────────────────────

def xml_to_csv():
    rows = []

    for folder in DATA_FOLDERS:
        base = os.path.basename(folder).upper()
        if base.startswith("CNTRL"):
            grp = "Control"
        elif base.startswith("CTRCD"):
            grp = "CTRCD"
        else:
            raise ValueError(f"Unrecognized data folder: {folder!r}")

        pattern = os.path.join(folder, "**", "*.xml")
        for xml_path in glob.glob(pattern, recursive=True):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot().find("./RestingECG") or tree.getroot()
            except ET.ParseError:
                continue

            row = {"Group": grp}

            # 1) flat metadata
            for col, xpath in METADATA_FIELDS.items():
                node = root.find(xpath)
                row[col] = node.text.strip() if (node is not None and node.text) else None

            # 2) intervals
            for col, xpath in INTERVAL_FIELDS.items():
                node = root.find(xpath)
                row[col] = node.text.strip() if (node is not None and node.text) else None

            # 3) abnormal flag
            row["Abnormal"] = is_abnormal_ecg(root)

            # 4) find the full-length sample count and decode only those leads
            all_ld = root.findall(".//Waveform/LeadData")
            counts = [int(ld.findtext("LeadSampleCountTotal", "0") or 0) for ld in all_ld]
            max_count = max(counts, default=0)

            for ld in all_ld:
                cnt = int(ld.findtext("LeadSampleCountTotal", "0") or 0)
                if cnt != max_count:
                    continue

                lead_id, raw_sig = decode_lead(ld)
                sig250 = normalize_to_250hz(raw_sig)
                row[f"Lead_{lead_id}"] = json.dumps(sig250)

            rows.append(row)

    # build DataFrame & save
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Wrote {len(df)} rows × {len(df.columns)} cols to '{OUTPUT_CSV}'")


if __name__ == "__main__":
    xml_to_csv()

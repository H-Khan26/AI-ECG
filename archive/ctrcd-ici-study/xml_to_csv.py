

def base64_to_numerical(base64_string):
    """
    Converts Base64-encoded raw signal data (little-endian 16-bit integers) into numerical values.
    """
    try:
        # Decode Base64 string into raw bytes
        raw_bytes = base64.b64decode(base64_string)

        # Debugging: Print byte length and sample bytes
        print(f"Decoded Byte Length: {len(raw_bytes)}")
        print(f"Sample Bytes (Hex): {raw_bytes[:10].hex()}...")  # Show first 10 bytes

        # Ensure byte length is a multiple of 2 (16-bit samples)
        if len(raw_bytes) % 2 != 0:
            print("Warning: Byte length is not a multiple of 2. Trimming last byte.")
            raw_bytes = raw_bytes[:len(raw_bytes) // 2 * 2]

        # Convert bytes to 16-bit signed integers (Little-Endian)
        num_samples = len(raw_bytes) // 2
        numerical_values = struct.unpack(f'<{num_samples}h', raw_bytes)

        return np.array(numerical_values)

    except Exception as e:
        print(f"Error: {e}")
        return np.array([])

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



def extract_ecg_features(lead_I, lead_II, sampling_rate=250):
    """
    From two 1-D NumPy arrays (Lead I & Lead II):
      • median heart rate (bpm)
      • median QRS duration (ms)
      • median QT interval (ms)
      • T-wave frontal-plane axis (°)
    """
    # 1) clean & find R-peaks on Lead II
    clean = nk.ecg_clean(lead_II, sampling_rate=sampling_rate)
    peaks, _ = nk.ecg_peaks(clean, sampling_rate=sampling_rate)

    # 2) delineate P/QRS/T
    delineate = nk.ecg_delineate(clean,
                                 peaks,
                                 sampling_rate=sampling_rate,
                                 method="dwt")

    Q_on  = delineate["ECG_QRS_Onsets"]
    Q_off = delineate["ECG_QRS_Offsets"]
    T_off = delineate["ECG_T_Offsets"]
    T_pk  = delineate["ECG_T_Peaks"]

    # helper: samples → milliseconds
    to_ms = lambda arr: (arr / sampling_rate) * 1000

    # 3) heart-rate series → median
    hr_series = nk.ecg_rate(peaks, sampling_rate=sampling_rate)
    hr_bpm    = float(np.median(hr_series)) if len(hr_series)>0 else np.nan

    # 4) durations → median
    qrs_ms = float(np.median(to_ms(Q_off - Q_on))) if len(Q_on)>0 and len(Q_off)>0 else np.nan
    qt_ms  = float(np.median(to_ms(T_off - Q_on))) if len(Q_on)>0 and len(T_off)>0 else np.nan

    # 5) T-axis from first T-peak vector (I vs II)
    if len(T_pk) > 0:
        idx = T_pk[0]
        v1, v2 = lead_I[idx], lead_II[idx]
        theta = np.degrees(np.arctan2(v2, v1))
        if theta < 0:
            theta += 360
    else:
        theta = np.nan

    return {
        "HeartRate_bpm": hr_bpm,
        "QRS_ms":       qrs_ms,
        "QT_ms":        qt_ms,
        "T_Axis_deg":   float(theta),
    }

def is_v_paced(xml_path):

    """
    Detects ventricular pacing in an ECG XML by:
      1) Checking any <QRSTimesTypes>/<QRS>/<Type> == "P"
      2) Falling back to searching diagnosis text for "paced"
    """
    tree = ET.parse(xml_path)
    root = tree.getroot().find("./RestingECG") or tree.getroot()

    # 1) Look for paced QRS types
    for qrs in root.findall(".//QRSTimesTypes/QRS"):
        t = qrs.findtext("Type", "").strip().upper()
        if t == "P":
            return True

    # 2) Fallback: free-text mention of "paced"
    for stmt in root.findall(".//Diagnosis/DiagnosisStatement/StmtText"):
        if stmt.text and "paced" in stmt.text.lower():
            return True

    return False

def remove_v_paced(source_folder):
    """
    Moves all V-paced XML files from source_folder into source_folder/v_paced_removed/
    """
    backup_dir = os.path.join(source_folder, "v_paced_removed")
    os.makedirs(backup_dir, exist_ok=True)

    xml_files = glob.glob(os.path.join(source_folder, "*.xml"))
    removed = 0

    for xml_path in xml_files:
        try:
            if is_v_paced(xml_path):
                basename = os.path.basename(xml_path)
                dest = os.path.join(backup_dir, basename)
                shutil.move(xml_path, dest)
                print(f"Removed paced ECG: {basename}")
                removed += 1
        except Exception as e:
            print(f"Error processing {xml_path}: {e}")

    print(f"\nDone: moved {removed} paced file(s) to\n  {backup_dir}")


def xml_to_ecg_array(xml_path, target_sr=250, duration_s=10):
    """
    Reads one 12-lead ECG XML and returns a NumPy array shaped (T,12,1):
       T = target_sr * duration_s  (e.g. 250*10=2500)
       12 leads in order: I, II, III, aVR, aVL, aVF, V1–V6
       1 channel dim for compatibility.
    Amplitude is in millivolts.
    """
    # ─── 1) Parse XML & find full-strip leads ─────────────────────────────
    tree = ET.parse(xml_path)
    root = tree.getroot().find("./RestingECG") or tree.getroot()
    lead_nodes = root.findall("./Waveform/LeadData")

    # pick the block with the largest sample count
    sample_counts = [int(ld.findtext("LeadSampleCountTotal","0")) for ld in lead_nodes]
    max_count = max(sample_counts)
    full_nodes = [ld for ld,sc in zip(lead_nodes,sample_counts) if sc==max_count]

    # ─── 2) Decode each raw lead into an mV array ──────────────────────────
    raw_leads = {}
    for ld in full_nodes:
        lid   = ld.findtext("LeadID","").strip()           # e.g. "I","II","V1",…
        units = float(ld.findtext("LeadAmplitudeUnitsPerBit","1") or 1.0)
        b64   = ld.findtext("WaveFormData","").strip()
        raw   = base64.b64decode(b64)

        # little-endian 16-bit signed
        n      = len(raw)//2
        vals   = struct.unpack(f"<{n}h", raw)
        arr    = np.array(vals, dtype=float) * units     # now in mV

        raw_leads[lid] = arr

    # ─── 3) Reconstruct the 4 derived limb leads ───────────────────────────
    I  = raw_leads["I"]
    II = raw_leads["II"]
    raw_leads["III"] = II - I
    raw_leads["aVR"] = -(I + II)/2
    raw_leads["aVL"] =  I - II/2
    raw_leads["aVF"] =  II - I/2

    # ─── 4) Stack in the specified order ──────────────────────────────────
    order = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
    sig = np.stack([raw_leads[l] for l in order], axis=1)  # shape (N_orig, 12)

    # ─── 5) Resample to (target_sr * duration_s) samples ─────────────────
    T_target = int(target_sr * duration_s)
    sig_res  = resample(sig, T_target, axis=0)

    # ─── 6) Add channel dim and return ───────────────────────────────────
    return sig_res[..., np.newaxis]   # shape = (T_target, 12, 1)

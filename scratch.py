
rates = Counter()
durations = []

for xml in glob.glob("EKG_DATA_CTRL/**/*.xml", recursive=True):
    tree = ET.parse(xml)
    r = tree.getroot().find(".//ECGSampLeBase").text
    e = tree.getroot().find(".//ECGSampleExponent").text
    fs = int(r) * (10 ** int(e))
    sc = int(tree.getroot().find(".//LeadSampleCountTotal").text)
    rates[fs] += 1
    durations.append(sc / fs)

print("Sampling rates seen:", rates)
print("Min/Max duration (s):", min(durations), max(durations))
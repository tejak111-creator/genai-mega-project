import pandas as pd
from app.rca.analysis import detect_cpu_spikes

def test_detect_cpu_spikes_basic():
    df = pd.DataFrame({
        "cpu" : [50,85,90,30],
        "service" : ["a","b","c","d"]
    })

    result = detect_cpu_spikes(df, threshold=80)

    assert len(result) == 2
    assert all(result["cpu"] > 80)

def test_detect_cpu_spikes_no_spikes():
    df=pd.DataFrame({
        "cpu":[10,20,30]
    })

    result = detect_cpu_spikes(df, threshold=80)
    assert len(result) == 0
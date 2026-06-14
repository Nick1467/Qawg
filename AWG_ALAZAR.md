# AWG5208 + ATS9371 統一介面

`QAWG/awg_alazar.py` 用 `AWGAlazar` 保存兩台儀器的 sampling rate，並負責
AWG 播放、Alazar trigger、DMA acquisition 和 IQ demodulation。

## 建立連線

```python
from QAWG.awg_alazar import AWGAlazar

experiment = AWGAlazar.connect(
    "TCPIP0::192.168.10.171::inst0::INSTR",
    awg_sample_rate_hz=2.5e9,
    alazar_sample_rate_hz=1e9,
    tone_frequency_hz=50e6,
    trigger_delay_s=100e-9,
    num_averages=1000,
    acquire_window_ns=1200,
    integrate_time_s=1e-6,
    adc_channel="CHA",
    moving_average_time_s=20e-9,
)
```

必要的 Alazar 量測參數是：

- `trigger_delay_s`: 收到外部 trigger 後，延遲多久開始記錄。
- `num_averages`: AWG trigger 與 Alazar record 的重複次數。
- `acquire_window_ns`: 每個 trigger 要保存的完整 ADC record。
- `integrate_window_ns=(start_ns, stop_ns)`: `acquire()` 用來計算
  單一 IQ 點的區間。
- `adc_channel`: 接收資料的 Alazar input，可使用 `"CHA"`、`"CHB"`、
  `0` 或 `1`。其中 `0` 是 CHA、`1` 是 CHB。

Raw voltage 與 downconverted IQ 都保留完整 `acquire_window_ns`。
`integrate_window_ns` 必須位於 acquire window 內，時間零點是 Alazar
開始記錄的位置。`moving_average_time_s` 是 `acquire_decimate()` 的
boxcar 寬度。

ATS9371 record 長度會向上對齊到 128 samples 的倍數，因此實際取得的
資料可能比 `acquire_window_ns` 稍長，但不會比要求的 window 短。

## 時間轉 sample

```python
experiment.ns2cycles(100)              # 250 DAC cycles at 2.5 GS/s
experiment.ns2cycles(100, inst="dac")  # 250 AWG DAC cycles
experiment.ns2cycles(100, inst="adc")  # 100 Alazar ADC cycles

experiment.cycles2ns(250)              # 100 ns using AWG sampling rate
experiment.cycles2ns(100, inst="adc")  # 100 ns using Alazar sampling rate
```

`inst` 預設為 `"dac"`，因此沒有指定時會使用 AWG sampling rate。
使用 `"adc"` 時則使用 Alazar sampling rate。

## 兩種 acquisition

取得保留時間軸的 moving-average IQ 波形：

```python
time_s, average_iq = experiment.acquire_decimate()
```

取得 integration window 內平均後的一個 IQ 點，以及每個 shot 的完整
downconverted IQ trace：

```python
iq, downconverted_iq = experiment.acquire()
```

`downconverted_iq.shape` 為 `(num_averages, acquire_samples)`。之後要計算
每個 shot 隨 integration time 增長的 trajectory，可以直接使用：

```python
trajectory_iq = np.cumsum(downconverted_iq, axis=1)
trajectory_iq /= np.arange(1, downconverted_iq.shape[1] + 1)
```

每次量測的 debug 資料保存在：

```python
experiment.last_raw_codes
experiment.last_records_volts
experiment.last_downconverted_iq
experiment.last_shot_iq
experiment.last_time_s
```

查看 ADC channel、LSB、DC offset 與 code range：

```python
experiment.capture_diagnostics()
```

目前 ATS9371 driver 固定使用 `+/-400 mV` input range。12-bit ADC 的
一個 code 約為 `0.195 mV`，因此接近或小於此振幅的訊號，在單次 raw
record 中會呈現明顯量化階梯。

使用完畢後：

```python
experiment.close()
```

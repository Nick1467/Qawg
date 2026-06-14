# AWG Phase Sweep 與 IQ 不連續問題

## 現象

掃描 AWG carrier phase 時，理論上 demodulation 後的 IQ 點應該沿著圓周連續旋轉：

```text
AWG phase 增加 11.25 deg
-> IQ angle 也增加約 11.25 deg
```

原本測量 32 個 phase 點時，在下列區域出現明顯跳變：

```text
56.25 deg -> 67.5 deg
236.25 deg -> 247.5 deg
```

跳變每隔 180 deg 重複一次，表示問題不是隨機雜訊，而是與 sine carrier
的正負半週或 zero crossing 有關。

## 根本原因

原本 `marker()` 使用已完成 carrier modulation 的 analog waveform 判斷 pulse
起點：

```python
analog = envelope * np.sin(2 * np.pi * fc * time + phase)
```

接著用相對於 waveform peak 的 threshold 找第一個 active sample：

```python
active = np.abs(analog) >= threshold
```

但是 carrier phase 改變時，pulse 開頭的 sine 值也會改變。第一個超過
threshold 的 sample 因此會隨 phase 前後移動。

結果是：

1. AWG phase 被改變。
2. Marker rising edge 也跟著移動。
3. ATS9371 使用 marker rising edge 作為每筆 record 的時間原點。
4. Digital demodulation reference 仍從 record 的 sample 0 開始。
5. Trigger timing 的移動被額外轉換成 IQ phase。

因此實驗並不是只改變 AWG phase，而是同時改變：

```text
signal phase + acquisition time origin
```

當 threshold 找到的 carrier half-cycle 切換時，marker 起點會跳到另一個
zero-crossing 分支，IQ 圖上就會出現不連續。

## 為什麼每 180 度重複

Marker 判斷使用 `abs(waveform)`。Sine 波相差 180 deg 時：

```text
sin(theta + pi) = -sin(theta)
abs(sin(theta + pi)) = abs(sin(theta))
```

所以相同的 marker timing 錯誤會每 180 deg 重複一次。這與量測中兩個跳變
區域相隔約 180 deg 一致。

## Driver 修正

Driver 現在同時保存兩種資料：

```text
analog waveform:
    envelope * carrier

activity waveform:
    未調變的 envelope
```

`marker()` 改為使用完整、未調變的 activity envelope 判斷 pulse 起訖：

```python
marker_name = awg.marker(
    waveform_ch=3,
    marker_ch=1,
)
```

Activity envelope 仍保留：

- timeline 的 leading delay
- pulse duration
- `delay()` 與 `delay_auto()`
- 所有 channel 的共同總長度

但它不受下列參數影響：

- carrier frequency
- carrier phase
- carrier zero crossing

因此 phase sweep 過程中 marker rising edge 保持在完全相同的 sample。

## 實機驗證

修正後使用 AWG5208 CH3 輸出 50 MHz Gaussian-square，CH1 marker 觸發
ATS9371，量到：

```text
Programmed phase    Measured IQ angle
56.25 deg           -19.11 deg
67.50 deg            -8.30 deg
78.75 deg             3.33 deg
90.00 deg            14.33 deg
```

原本的跳變區域已變成連續旋轉。完整 phase fit 結果：

```text
phase slope          0.999316
fixed phase offset  -75.382 deg
maximum residual     0.975 deg
AWG error             0, "No error"
```

`phase slope` 接近 1，表示 AWG 每增加 1 deg，IQ angle 也增加約 1 deg。
固定的 `-75.382 deg` offset 是 cable delay、AWG sine convention 與
demodulation reference 共同造成的固定參考相位，不代表 phase sweep 錯誤。

## 建議的驗證程式

不要只看 IQ 圖是否像圓，也要計算 phase transfer：

```python
averaged_points = np.asarray(averaged_points)
measured_angles = np.unwrap(np.angle(averaged_points))

phase_slope, phase_offset = np.polyfit(
    phases,
    measured_angles,
    1,
)

phase_residual_deg = np.rad2deg(
    measured_angles
    - (phase_slope * phases + phase_offset)
)

radius = np.abs(averaged_points)

print("Phase slope:", phase_slope)
print("Fixed phase offset (deg):", np.rad2deg(phase_offset))
print("Maximum phase residual (deg):", np.max(np.abs(phase_residual_deg)))
print(
    "Radius variation (%):",
    np.std(radius) / np.mean(radius) * 100,
)
```

理想結果：

```text
phase slope ~= 1
phase residual 小且沒有突然跳變
radius variation 小
```

## 使用注意事項

1. 更新 driver 後需重新啟動 Jupyter kernel，否則 notebook 仍會使用記憶體中的舊 class。
2. 不要用 carrier-modulated waveform 的 zero crossing 產生 acquisition trigger。
3. Marker 應依 envelope 或明確的時間窗口產生。
4. AWG 與 ATS9371 共用 10 MHz reference 只能穩定長期頻率關係；它不能修正由 marker timing 漂移造成的 record time-origin 錯誤。
5. 比較 phase 時應使用 coherent complex average，而不是先取每個 shot 的 magnitude。


# AWG5208 Sequence List 與 T1 實驗設計

## 結論

T1 sweep 不應該採用以下流程：

```text
設定 delay 1
-> 重新建立 waveform
-> 上傳
-> 擷取
-> 設定 delay 2
-> 再上傳
-> 再擷取
```

這會讓網路傳輸、WFMX 建立、AWG waveform loading 和 channel assignment
變成實驗的主要時間。

第一版建議：

```text
一次產生所有 T1 delay point
-> 打包成一個 SEQX
-> 上傳一次
-> AWG sequence 自動播放所有 delay
-> ATS9371 連續接收所有 marker records
```

因此答案是：**T1 應該使用 sequence list**。

但 sequence list 不一定代表完全不產生每個 delay 的 waveform。第一版仍可為
每個 delay 建立一組 waveform，只是所有 waveform 會被一次打包與上傳，不會
在 sweep 過程中重複傳輸。

## AWG 的兩層資料

AWG5208 將資料分成兩層：

```text
Waveform list
    保存實際 analog waveform 與 marker samples

Sequence list
    保存每個 step 要播放哪個 waveform、重複幾次、
    是否等待 trigger、下一個 step 和 event jump
```

QCoDeS 的 AWG70000/AWG5208 driver 也是這樣處理：

1. 每個 sequence element 和 channel 先建立 WFMX。
2. 使用 SML 描述 sequence steps。
3. 將 WFMX、SML、setup.xml 和 userNotes.txt 打包成 SEQX。
4. 將 SEQX 上傳到 AWG。
5. 載入後，sequence 出現在 AWG sequence list。
6. 將 sequence track 指派給實體 channel。

QCoDeS sequence step 支援：

```text
trig_wait
nrep
event_jump
event_jump_to
goto
flags
```

其中：

- `trig_wait=0`：直接播放。
- `trig_wait=1`：等待 Trigger A。
- `trig_wait=2`：等待 Trigger B。
- `nrep=1`：播放一次。
- `nrep=N`：同一個 step 播放 N 次。
- `nrep=0`：無限重複。
- `goto=0`：進入下一個 step。
- `goto=N`：播放後跳到指定 step。

## 建議的 T1 waveform

假設：

- CH2：Qubit pulse
- CH3：Readout pulse
- CH1 Marker 1：ATS9371 trigger
- Delay points：`t1_delays`
- 每個 shot 的固定總長度：`shot_duration`

每個 delay point 產生一個完整、固定長度的 shot：

```text
CH2 | pi pulse |---------------- zero ----------------------|

CH3 |----------- delay ---------| readout |------ zero -----|

CH1 |----------------------------| marker  |-----------------|
```

所有 step 應使用相同的總長度：

```python
shot_duration >= max(t1_delays) + readout_duration + reset_margin
```

這樣每個 shot 的 repetition period 固定，不會因為 delay 改變而改變 AWG
輸出頻率或 ATS trigger cadence。

## Sequence 結構

例如有五個 delay：

```python
t1_delays = [
    0,
    100e-9,
    200e-9,
    500e-9,
    1e-6,
]
```

Sequence 可以是：

```text
Step 1: T1 delay = 0 ns
Step 2: T1 delay = 100 ns
Step 3: T1 delay = 200 ns
Step 4: T1 delay = 500 ns
Step 5: T1 delay = 1000 ns
Step 5 goto Step 1
```

每個 step 都包含：

- CH2 對應的 qubit waveform
- CH3 對應的 readout waveform
- CH1 對應的 zero analog waveform 和 marker

## Averaging 的兩種順序

### 方法一：每個 delay 連續平均

```text
delay 0:     shot 1, shot 2, ..., shot N
delay 100ns: shot 1, shot 2, ..., shot N
delay 200ns: shot 1, shot 2, ..., shot N
```

每個 step 設：

```text
nrep = number_of_averages
```

ATS records 排列：

```python
records.shape == (
    number_of_delays * number_of_averages,
    samples_per_record,
)

records = records.reshape(
    number_of_delays,
    number_of_averages,
    samples_per_record,
)
```

優點是資料排列直觀。缺點是慢速 gain drift 或 qubit frequency drift 可能使不同
delay point 在不同時間取得。

### 方法二：交錯 delay points

```text
average 1: delay 0, delay 100ns, delay 200ns, ...
average 2: delay 0, delay 100ns, delay 200ns, ...
```

每個 step：

```text
nrep = 1
```

最後一個 step：

```text
goto = 1
```

ATS records 排列：

```python
records = records.reshape(
    number_of_averages,
    number_of_delays,
    samples_per_record,
)
```

再沿著 average axis 平均：

```python
average_records = records.mean(axis=0)
```

**建議 T1 使用交錯方式**，因為所有 delay point 會在相近時間被量測，較不受
慢速 drift 影響。

## ATS9371 擷取流程

目前的 hardware order 可以保留：

```text
1. 停止 AWG
2. Arm ATS9371 DMA
3. Start ATS capture
4. 啟動 AWG sequence
5. AWG 每個 step 的 marker 觸發一筆 ATS record
6. ATS 收滿 number_of_delays * number_of_averages records
7. 停止 AWG
8. Reshape records
9. Demodulate 和 average
```

總 records 數：

```python
number_of_records = (
    number_of_delays
    * number_of_averages
)
```

Marker 必須每個 shot 只產生一次 rising edge，並與 readout envelope 對齊。

## 為什麼第一版不先拆成 pulse、delay、readout assets

理論上可以建立可重用元件：

```text
pi pulse waveform
idle waveform
readout waveform
marker waveform
```

再用 sequence repeat 組合不同 delay：

```text
pi -> idle repeated N times -> readout
```

但 AWG5200 的 WFMX waveform 有最小 sample 數限制。目前 driver 使用的限制是
2400 samples。在 2.5 GS/s：

```text
2400 / 2.5e9 = 960 ns
```

如果只依靠重複一個最短 idle waveform，delay resolution 最多只能做到
960 ns，無法直接表示常見的 10 ns、20 ns 或 100 ns T1 間隔。

要使用 component reuse，還需要：

- coarse idle repeat
- fine remainder waveform
- subsequence
- 每個 channel 的 step boundary 對齊
- 對 sequence transition timing 做實機驗證

因此它比較適合作為第二階段最佳化，而不是第一版 T1 driver。

## 記憶體估算

即使第一版為每個 delay 建立完整 waveform，通常仍可接受。

例如：

```text
sample rate      = 2.5 GS/s
shot duration    = 5 us
samples/shot     = 12,500
delay points     = 101
channels         = 3
```

Analog float32 加上一個 marker 約 5 bytes/sample：

```text
12,500 * 101 * 3 * 5
= 約 18.9 MB
```

這遠低於 AWG5200 每 channel 的 waveform memory。真正需要避免的是每個 point
都透過網路重新上傳，而不是這十幾 MB 的預先編譯資料。

## Driver 建議 API

建議新增純資料結構：

```python
sequence = t1_sequence(
    qubit=qubit,
    readout=readout,
    delays_s=t1_delays,
    marker_ch=1,
    total_duration_s=5e-6,
    ordering="interleaved",
)
```

再由 hardware boundary 一次上傳：

```python
sequence_name = awg.upload_sequence(
    sequence,
    amplitude_vpp={
        1: 0.5,
        2: 0.5,
        3: 0.5,
    },
    name="t1_experiment",
)
```

執行：

```python
records = run_sequence_acquisition(
    awg=awg,
    ats_api=ats_api,
    ats_board=ats_board,
    sequence_name=sequence_name,
    number_of_delays=len(t1_delays),
    number_of_averages=1000,
)
```

`upload_sequence()` 應負責：

1. 驗證每個 step 的 channel 數與 waveform 長度。
2. 建立每個 WFMX。
3. 建立 SML sequence description。
4. 打包 SEQX。
5. 一次上傳並載入。
6. 將每個 sequence track 指派到對應 channel。

## 實作順序

第一階段：

```text
每個 delay 一個完整 waveform step
一次 SEQX upload
interleaved playback
ATS records reshape
```

第二階段：

```text
支援 step repeat、trigger wait、goto
```

第三階段：

```text
subsequence
coarse delay repeat + fine remainder
waveform deduplication
```

第一階段已經可以消除 T1 sweep 中反覆上傳 waveform 的主要效能問題。


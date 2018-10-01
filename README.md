Simple TensorFlow Speed Test
===

__benchmark_single_DATAOnGPU__

| Operating System | 1080 TI x 1 |
|-------|------------|
| Windows 10 | |
| Ubuntu 18.04 | |


```
python benchmark_single_DataOnGPU.py --device_id=0
python benchmark_single_DataOnGPU.py --device_id=1
python benchmark_single_DataOnGPU.py --device_id=2
python benchmark_single_DataOnGPU.py --device_id=3
```


__benchmark_single_CPU2GPU__

| Operating System | 1080 TI x 1 |
|-------|------------|
| Windows 10 | |
| Ubuntu 18.04 | |

```
python benchmark_single_CPU2GPU.py --device_id=0
python benchmark_single_CPU2GPU.py --device_id=1
python benchmark_single_CPU2GPU.py --device_id=2
python benchmark_single_CPU2GPU.py --device_id=3
```

__benchmark_single_TFDataset__

| Operating System | 1080 TI x 1 |
|-------|------------|
| Windows 10 | |
| Ubuntu 18.04 | |

```
python benchmark_single_TFDataset.py --device_id=0
python benchmark_single_TFDataset.py --device_id=1
python benchmark_single_TFDataset.py --device_id=2
python benchmark_single_TFDataset.py --device_id=3
```

__benchmark_ps_DATAOnGPU__

| Operating System | 1080 TI x 1 | 1080 TI x 2 | 1080 TI x 3 | 1080 TI x 4 |
|-------|------------|------------|------------|------------|
| Windows 10 | | | | |
| Ubuntu 18.04 | | | | |

```
python benchmark_ps_DataOnGPU.py --device_list=0
python benchmark_ps_DataOnGPU.py --device_list=0,1
python benchmark_ps_DataOnGPU.py --device_list=0,1,2
python benchmark_ps_DataOnGPU.py --device_list=0,1,2,3
```

__benchmark_ps_CPU2GPU__

| Operating System | 1080 TI x 1 | 1080 TI x 2 | 1080 TI x 3 | 1080 TI x 4 |
|-------|------------|------------|------------|------------|
| Windows 10 | | | | |
| Ubuntu 18.04 | | | | |

```
python benchmark_ps_CPU2GPU.py --device_list=0
python benchmark_ps_CPU2GPU.py --device_list=0,1
python benchmark_ps_CPU2GPU.py --device_list=0,1,2
python benchmark_ps_CPU2GPU.py --device_list=0,1,2,3
```

__benchmark_ps_TFDataset__

| Operating System | 1080 TI x 1 | 1080 TI x 2 | 1080 TI x 3 | 1080 TI x 4 |
|-------|------------|------------|------------|------------|
| Windows 10 | | | | |
| Ubuntu 18.04 | | | | |

```
python benchmark_ps_TFDataset.py --device_list=0
python benchmark_ps_TFDataset.py --device_list=0,1
python benchmark_ps_TFDataset.py --device_list=0,1,2
python benchmark_ps_TFDataset.py --device_list=0,1,2,3
```

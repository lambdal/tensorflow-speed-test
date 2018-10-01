Simple TensorFlow Speed Test
===

__benchmark_single_DATAOnGPU__

| Operating System | 1080 TI x 1 |
|-------|------------|
| Windows 10, i7-6850| 950 |
| Ubuntu 18.04 i9-7900x| 988 |


```
python benchmark_single_DataOnGPU.py --device_id=0
python benchmark_single_DataOnGPU.py --device_id=1
python benchmark_single_DataOnGPU.py --device_id=2
python benchmark_single_DataOnGPU.py --device_id=3
```


__benchmark_single_CPU2GPU__

| Operating System | 1080 TI x 1 |
|-------|------------|
| Windows 10 i7-6850| 533 |
| Ubuntu 18.04 i9-7900x| 560 |

```
python benchmark_single_CPU2GPU.py --device_id=0
python benchmark_single_CPU2GPU.py --device_id=1
python benchmark_single_CPU2GPU.py --device_id=2
python benchmark_single_CPU2GPU.py --device_id=3
```

__benchmark_single_TFDataset__

| Operating System | 1080 TI x 1 |
|-------|------------|
| Windows 10, i7-6850| 730 |
| Ubuntu 18.04 i9-7900x| 800 |

```
python benchmark_single_TFDataset.py --device_id=0
python benchmark_single_TFDataset.py --device_id=1
python benchmark_single_TFDataset.py --device_id=2
python benchmark_single_TFDataset.py --device_id=3
```

__benchmark_ps_DATAOnGPU__

| Operating System | 1080 TI x 1 | 1080 TI x 2 | 1080 TI x 3 | 1080 TI x 4 |
|-------|------------|------------|------------|------------|
| Windows 10, i7-6850| 930 | 1882 | 2800 | 3673 |
| Ubuntu 18.04 i9-7900x| 950 | 1910 | 2800 | 3820 |

```
python benchmark_ps_DataOnGPU.py --device_list=0
python benchmark_ps_DataOnGPU.py --device_list=0,1
python benchmark_ps_DataOnGPU.py --device_list=0,1,2
python benchmark_ps_DataOnGPU.py --device_list=0,1,2,3
```

__benchmark_ps_CPU2GPU__

| Operating System | 1080 TI x 1 | 1080 TI x 2 | 1080 TI x 3 | 1080 TI x 4 |
|-------|------------|------------|------------|------------|
| Windows 10, i7-6850| 575 | 765 | 840 | 850 |
| Ubuntu 18.04 i9-7900x| 490 | 690 | 870 | 990 |

```
python benchmark_ps_CPU2GPU.py --device_list=0
python benchmark_ps_CPU2GPU.py --device_list=0,1
python benchmark_ps_CPU2GPU.py --device_list=0,1,2
python benchmark_ps_CPU2GPU.py --device_list=0,1,2,3
```

__benchmark_ps_TFDataset__

| Operating System | 1080 TI x 1 | 1080 TI x 2 | 1080 TI x 3 | 1080 TI x 4 |
|-------|------------|------------|------------|------------|
| Windows 10, i7-6850| 790 | 975 | 1131 | 1200 |
| Ubuntu 18.04 i9-7900x| 800 | 1450 | 1941 | 1950 |

```
python benchmark_ps_TFDataset.py --device_list=0
python benchmark_ps_TFDataset.py --device_list=0,1
python benchmark_ps_TFDataset.py --device_list=0,1,2
python benchmark_ps_TFDataset.py --device_list=0,1,2,3
```

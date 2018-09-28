Simple TensorFlow Speed Test
===

__benchmark_single_GPU_DATAOnGPU__

Referecen speed on 1080TI: 
Windows 10:
Ubuntu 18.04: 980 images/sec

```
python benchmark_single_GPU_DataOnGPU.py --device_id=0
python benchmark_single_GPU_DataOnGPU.py --device_id=1
python benchmark_single_GPU_DataOnGPU.py --device_id=2
python benchmark_single_GPU_DataOnGPU.py --device_id=3
```


__benchmark_single_GPU_CPU2GPU__

Referecen speed on 1080TI
Windows 10:
Ubuntu 18.04: 550 images/sec or 700 images/sec

```
python benchmark_single_GPU_CPU2GPU.py --device_id=0
python benchmark_single_GPU_CPU2GPU.py --device_id=1
python benchmark_single_GPU_CPU2GPU.py --device_id=2
python benchmark_single_GPU_CPU2GPU.py --device_id=3
```

__benchmark_ps_DATAOnGPU__

Referecen speed on 1080TI: 
Windows 10:
Ubuntu 18.04: 950 1900 2870 3800

```
python benchmark_ps_DataOnGPU.py --list_devices=0
python benchmark_ps_DataOnGPU.py --list_devices=0,1
python benchmark_ps_DataOnGPU.py --list_devices=0,1,2
python benchmark_ps_DataOnGPU.py --list_devices=0,1,2,3
```

__benchmark_ps_CPU2GPU__

Referecen speed on 1080TI: 
Windows 10:
Ubuntu 18.04: 480 700 870 1000

```
python benchmark_ps_CPU2GPU.py --list_devices=0
python benchmark_ps_CPU2GPU.py --list_devices=0,1
python benchmark_ps_CPU2GPU.py --list_devices=0,1,2
python benchmark_ps_CPU2GPU.py --list_devices=0,1,2,3
```

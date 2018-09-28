Simple TensorFlow Speed Test
===

__benchmark_single_GPU_DATAOnGPU__

Referecen speed on 1080TI: 
Windows 10:
Ubuntu 18.04:

```
python benchmark_single_GPU_DATAOnGPU.py --device_id=0
python benchmark_single_GPU_DATAOnGPU.py --device_id=1
python benchmark_single_GPU_DATAOnGPU.py --device_id=2
python benchmark_single_GPU_DATAOnGPU.py --device_id=3
```


__benchmark_single_GPU_CPU2GPU__

Referecen speed on 1080TI
Windows 10:
Ubuntu 18.04:

```
python benchmark_single_GPU_CPU2GPU.py --device_id=0
python benchmark_single_GPU_CPU2GPU.py --device_id=1
python benchmark_single_GPU_CPU2GPU.py --device_id=2
python benchmark_single_GPU_CPU2GPU.py --device_id=3
```

__benchmark_ps_DATAOnGPU__

Referecen speed on 1080TI: 
Windows 10:
Ubuntu 18.04:

```
python benchmark_ps_GPU_DATAOnGPU.py --device_id=0
python benchmark_ps_GPU_DATAOnGPU.py --device_id=0,1
python benchmark_ps_GPU_DATAOnGPU.py --device_id=0,1,2
python benchmark_ps_GPU_DATAOnGPU.py --device_id=0,1,2,3
```

__benchmark_ps_CPU2GPU__

Referecen speed on 1080TI: 
Windows 10:
Ubuntu 18.04:

```
python benchmark_ps_GPU_CPU2GPU.py --device_id=0
python benchmark_ps_GPU_CPU2GPU.py --device_id=0,1
python benchmark_ps_GPU_CPU2GPU.py --device_id=0,1,2
python benchmark_ps_GPU_CPU2GPU.py --device_id=0,1,2,3
```

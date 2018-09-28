Simple TensorFlow Speed Test
===

__Training on a single GPU__

Referecen speed on 1080TI: 530 images/sec
```
python benchmark_single_GPU.py --device_id=0
python benchmark_single_GPU.py --device_id=1
python benchmark_single_GPU.py --device_id=2
python benchmark_single_GPU.py --device_id=3
```


__Traning on a single GPU with parameter server setting__

Referecen speed on 1080TI: 550 images/sec
```
python benchmark_ps.py --list_devices=0
python benchmark_ps.py --list_devices=1
python benchmark_ps.py --list_devices=2
python benchmark_ps.py --list_devices=3
```

__Traning on a two GPUs with parameter server setting__

Referecen speed on 1080TI: 950 images/sec
```
python benchmark_ps.py --list_devices=0,1
```

__Traning on a three GPUs with parameter server setting__

Referecen speed on 1080TI: 1100 images/sec
```
python benchmark_ps.py --list_devices=0,1,2
```

__Traning on a four GPUs with parameter server setting__

```
python benchmark_ps.py --list_devices=0,1,2,3
```

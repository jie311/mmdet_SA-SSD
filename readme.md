## Experiments

### One Class: Car
||branch|Car AP@0.70, 0.70, 0.70|
|--|--|--|
|No AuxNet|no_aux|89.93, 79.83, 79.04|
|SA-SSD AuxNet|main|90.01, 86.23, 79.33|
|Our AuxNet|enhanced|89.96, 86.12, 79.24|


### Three Class: Car Pedestrian Cyclist
| branch||Car AP@0.70, 0.70, 0.70| Pedestrian AP@0.50, 0.50, 0.50|Cyclist AP@0.50, 0.50, 0.50|
|--|--|--|--|--|
|SA-SSD AuxNet|	main |89.85, 79.69, 78.97|52.75, 44.59, 43.75|67.90, 61.47, 60.33|
|Our AuxNet|enhanced|89.81, 79.67, 78.91|54.01, 46.34, 39.34|64.57, 65.13, 63.51|


Tips:

1. Metric: AP in 3d version
2. Our AuxNet: Unknown Points -> 3 Structure Points  + Mean Point

The complete experimental record is in `SA-SSD辅助网络改进.xlsx`  [百度云盘 00学习>10_科研>3D目标检测>记录>SA-SSD]
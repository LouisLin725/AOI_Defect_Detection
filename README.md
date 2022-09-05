# AOI_Defect_Detection
This is the subject of one of the exercises on AIdea platform.

## Table of contents
* [Image Info](#image-info)
* [Image Classification Groups](#image-classification-groups)
* [Sources](#sources)
* [Training Models](#training-models)
* [Results](#results)
* [Hardware Equipment](#hardware-equipment)
* [How to run](#how-to-run)
* [Reference](#reference)

## Brief Introduction
Automated optical inspection (AOI) is a high speed and precise optical detection system, which utilizing the computer vision (CV) as a core of detection technique. This method has the better performance in optical detection rather than manual inspection using the optical instrument in the past. The application of AOI is worldwide. We can easily find its trace in several domains, such as R&D in high-tech industries , manufacturing quality control, as well as national defense...etc. Therefore, this project is to improve the efficacy of AOI through the data sicience.

## Image Info
1. Training data: 2528
2. Testing data : 10142
3. Image groups : 1 Normal + 5 Defects
4. Image size   : 512 * 512

Data can be downloaded through the [Sources](#sources) connection.

ps. The ratio of training and validation data is 4:1.

## Image Classification Groups
![image](https://user-images.githubusercontent.com/101628791/188354248-ec0cc3fa-fe34-46b9-a701-8edf507469d6.png)

## Sources
The open competition on AIdea platform. 

Connection: https://aidea-web.tw/topic/285ef3be-44eb-43dd-85cc-f0388bf85ea4

## Training Models
VGG16_revised

LeNet5

## Results
| Model | Training Accuracy    | Validation Accuracy    | Testing Accuracy
| :---:   | :---: | :---: | :---: |
| VGG16_revised | 96.78 %   | 96.25 %   | 95.71 %   |
| LeNet5 | 97.57 %   | 98.22 %   | 96.05 %   |

## Hardware Equipment
CPU: Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz   2.81 GHz

RAM: 12.0 GB (4+8)

GPU: NVIDIA GeForce GTX 1050 / Intel(R) HD Graphics 630

GPU Memory: 2GB GDDR5 (128-bit)

## How to run
1. You can directly run the **train.py** file so that the model will be trained and exported.
2. The **submit.py** file can help you using the trained model to produce the test.csv, which is matched to the requirement of the AIdea.

## Reference
Methods providing
https://github.com/hcygeorge/aoi_defect_detection


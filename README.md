# Facial Expression Recognition by Global and ROI Features Fused CNN Combined with Attention Module

![image](https://github.com/tzu-tsen/code/assets/141349020/0c41c923-963e-4100-b6cc-6e6c354ef77e)

## Environment requirement
PyTorch 1.12.1、Python 3.10.8、Dlib 19.24.0

## Preparation
* Download [AffectNet](http://mohammadmahoor.com/affectnet/) dataset
* Download [RAF-DB](http://www.whdeng.cn/raf/model1.html) dataset
* Download [FERPlus](https://github.com/Microsoft/FERPlus) dataset
* Get ROI
  1. RAF-DB dataset use MTCNN get face
  2. Download [dlib 68 landmarks.dat](https://drive.google.com/file/d/1r-iq7F3u0VCQIedr43TD8iADF9UAxjTT/view?usp=drive_link), put it in the get_roi directory
  3. Go to get_roi folder
  4. face alignment
    ```
    python face_alignment.py 
    Enter annotation_folders (default:/autohome/user/tzutsen/database/RAF-DB/my/test): 
    Enter new_folders (default:test):
    ```
  5. Get dataset
    ```
    python face_roi.py 
    Enter annotation_folders (default:test/face_alignment): 
    Enter new_folders (default:face_roi/roi): 
    Enter error_folders (default:face_roi/error):
    ```
* Dataset schema
    Take train as an example
    * error is not get ROI image
    ```
    -- train
        |
        --original
        --error
        --roi
            |
            --eye
            --nose
            --mouth 
    ```

## Training
We provide the training code for AffectNet, RAF-DB And FERPlus.

* Go to folder：
    * Global-ROI Net：Go to global_roi folder
    * Global Net：Go to global folder

* For AffectNet dataset, run:
    ```
    python affectnet.py 
    Enter cuda (default:0): 
    Enter num_class (default:8): 
    Enter path_train (default:/autohome/user/tzutsen/database/AffectNet/train): 
    Enter path_test(val) (default:/autohome/user/tzutsen/database/AffectNet/test):
    ```

* For RAF-DB dataset, run:
    ```
    python rafdb.py
    Enter cuda (default:0): 
    Enter path_train (default:/autohome/user/tzutsen/database/RAF-DB/train): 
    Enter path_test (default:/autohome/user/tzutsen/database/RAF-DB/test): 
    ```

* For FERPlus dataset, run:
    ```
    python ferplus.py 
    Enter cuda (default:0): 
    Enter path (default:/autohome/user/tzutsen/database/FERPlus): 
    ```
	
## Test
We provide the training code for AffectNet, RAF-DB, FERPlus, CK+, JAFFE and SFEW 2.0.
| Dataset   | Global-ROI Net Accuracy (%)	| Global Net Accuracy (%) |	Accuracy (%)	| link |
| --- | --- | --- | --- | --- |
| AffectNet-7   | 65.71	| 64.95	| 65.49	| [download](https://drive.google.com/drive/folders/1bzdB46-KbbzEFQWyqQAcGh7a7RMq_zfd?usp=sharing) |
| AffectNet-8   | 61.85	| 61.25	| 62.12	| [download](https://drive.google.com/drive/folders/1Y1aDynHIoF60qIwmIvkqrepON-CctJNy?usp=sharing) |
| RAF-DB        | 88.81	| 87.22	| 87.65	| [download](https://drive.google.com/drive/folders/1a_MaEgAp4WjAbnrAx5pHFq2ckShj7ldp?usp=sharing) |
| FERPlus       | 86.25	| 85.50	| 85.19	| [download](https://drive.google.com/drive/folders/1oT66YrNeevVqzaFdmXXdeZYLLS0YPq_9?usp=sharing) |

Model trained on AffectNet database
| Dataset |	Global-ROI Net Accuracy (%) |	Global Net Accuracy (%) |	Accuracy (%) |
| --- | --- | --- | --- |
| CK+     |	87.72 |	89.76   |   90.02   |
| JAFFE   |	51.64 |	50.7    |	52.58   |
| SFEW2.0 |	38.39 |	37.8    |	37.73   |

Model trained on RAF-DB database
| Dataset |	Global-ROI Net Accuracy (%) |	Global Net Accuracy (%) |	Accuracy (%) |
| --- | --- | --- | --- |
| CK+     | 78.06	| 77.80	| 79.26 | 
| JAFFE	  | 45.54	| 47.42	| 46.95 |
| SFEW2.0	| 49.01	| 46.28	| 46.84 | 


* Go to acc folder

* For AffectNet7 dataset, run:
    ```
    python affectnet7.py 
    Enter path_model_roi (default:model/affectnet7/roi.path): 
    Enter path_model_global (default:model/affectnet7/global.path): 
    Enter db path (default:/autohome/user/tzutsen/database/AffectNet/test): 
    ```

* For AffectNet8 dataset, run:
    ```
    python affectnet8.py 
    Enter path_model_roi (default:model/affectnet8/roi.path): 
    Enter path_model_global (default:model/affectnet8/global.path): 
    Enter db path (default:/autohome/user/tzutsen/database/AffectNet/test): 
    ```

* For RAF-DB dataset, run:
    ```
    python rafdb.py 
    Enter path_model_roi (default:model/rafdb/roi.path): 
    Enter path_model_global (default:model/rafdb/global.path): 
    Enter db path (default:/autohome/user/tzutsen/database/RAF-DB/test): 
    ```

* For FERPlus dataset, run:
    ```
    python ferplus.py 
    Enter path_model_roi (default:model/ferplus/roi.path): 
    Enter path_model_global (default:model/ferplus/global.path): 
    Enter db path (default:/autohome/user/tzutsen/database/FERPlus): 
    ```

* For CK+ dataset, model trained on AffectNet database, run:
    ```
    python ck_affectnet8.py 
    Enter path_model_roi (default:model/affectnet8/roi.path): 
    Enter path_model_global (default:model/affectnet8/global.path): 
    ```

* For CK+ dataset, model trained on RAF-DB database, run:
    ```
    python ck_rafdb.py 
    Enter path_model_roi (default:model/rafdb/roi.path): 
    Enter path_model_global (default:model/rafdb/global.path): 
    Enter db path (default:/autohome/user/tzutsen/database/CK+/test): 
    ```

* For JAFFE dataset, model trained on AffectNet database, run:
    ```
    python jaffe_affectnet8.py 
    Enter path_model_roi (default:model/affectnet8/roi.path): 
    Enter path_model_global (default:model/affectnet8/global.path): 
    Enter db path (default:/autohome/user/tzutsen/database/JAFFE/test):
    ```

* For JAFFE dataset, model trained on RAF-DB database, run:
    ```
    python jaffe_rafdb.py 
    Enter path_model_roi (default:model/rafdb/roi.path): 
    Enter path_model_global (default:model/rafdb/global.path): 
    Enter db path (default:/autohome/user/tzutsen/database/JAFFE/test): 
    ```

* For SFEW 2.0 dataset, model trained on AffectNet database, run:
    ```
    python sfew_affectnet8.py 
    Enter path_model_roi (default:model/affectnet8/roi.path): 
    Enter path_model_global (default:model/affectnet8/global.path): 
    Enter db path (default:/autohome/user/tzutsen/database/SFEW2.0/test):
    ```

* For SFEW 2.0 dataset, model trained on RAF-DB database, run:
    ```
    python sfew_rafdb.py 
    Enter path_model_roi (default:model/rafdb/roi.path): 
    Enter path_model_global (default:model/rafdb/global.path): 
    Enter db path (default:/autohome/user/tzutsen/database/SFEW2.0/test):
    ``` 

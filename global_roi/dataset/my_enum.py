from enum import Enum

class EnumRun(Enum):
    train = 0
    val = 1
    test = 2
    testequalization = 3

class EnumRoi(Enum):
    original = 0
    eye = 1
    nose = 2
    mouth = 3

class EnumDirection(Enum):
    all = 0 #全部
    front = 1 #正臉
    profile = 2 # 側臉
    right = 3 #右側臉
    left = 4 #左側臉

class EnumType(Enum):
    original = 0 #原始
    align = 1 #對齊
    error = 2 #無法對齊和取roi
    roi = 3 
    re = 4 #roi+error
    
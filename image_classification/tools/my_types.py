r"""
为了简洁起见定义的类型别名, #TODO 暂时还是不用这个文件了，增加复杂性，其实用Op代替Optional就挺好 
"""
from typing import Optional, List, Tuple, Dict
r"""
命名规则：一个[]关系用_表示
"""
# 基本类型允许None
Op_str = Optional[str]
Op_int = Optional[int]
Op_float = Optional[float]
# 容器类型允许None
Op_List = Optional[List]
Op_List_str = Optional[List[str]]
Op_List_Op_str = Optional[List[Optional[str]]]
Op_List_int = Optional[List[int]]
Op_List_Op_int = Optional[List[Optional[int]]]
Op_List_float = Optional[List[float]]
Op_List_Op_float = Optional[List[Optional[float]]]
Op_List_Tuple = Optional[List[Tuple]]
Op_List_Op_Tuple = Optional[List[Optional[Tuple]]]
Op_Tuple = Optional[Tuple]
Dict = Optional[Dict]


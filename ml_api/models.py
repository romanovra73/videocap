from pydantic import BaseModel
import pydantic_numpy.typing as pnd
from pydantic_numpy import np_array_pydantic_annotated_typing
from numpydantic import NDArray
import numpy as np

class Img_Encoded(BaseModel):
    #data: np_array_pydantic_annotated_typing(data_type=np.int64, dimensions=1)
    data: bytes
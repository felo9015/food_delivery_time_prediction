"""Pydantic request/response schemas for the delivery-time prediction API.

The categorical fields below only accept the exact category values the
production model was actually trained on. Those values were read directly
from the fitted encoder in model_pipeline/artifacts/model.joblib -- not
guessed or hand-typed -- so the API can never send the model a category it
has never seen during training.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Weather(str, Enum):
    CLEAR = "Clear"
    FOGGY = "Foggy"
    RAINY = "Rainy"
    SNOWY = "Snowy"
    WINDY = "Windy"
    # "Unknown" is accepted as a normal, valid value, not just tolerated as a
    # fallback: this is exactly how a missing Weather value was handled during
    # training (model_pipeline/data_preprocessing.py imputes missing Weather
    # with the constant "Unknown" before encoding). A caller who does not have
    # this information can send "Unknown" explicitly instead of guessing a
    # real weather condition.
    UNKNOWN = "Unknown"


class TrafficLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    UNKNOWN = "Unknown"  # same reasoning as Weather.UNKNOWN above


class TimeOfDay(str, Enum):
    MORNING = "Morning"
    AFTERNOON = "Afternoon"
    EVENING = "Evening"
    NIGHT = "Night"
    UNKNOWN = "Unknown"  # same reasoning as Weather.UNKNOWN above


class VehicleType(str, Enum):
    BIKE = "Bike"
    CAR = "Car"
    SCOOTER = "Scooter"
    # No "Unknown" option here: EDA_report.md found zero missing values for
    # Vehicle_Type, so the training pipeline never saw, and never learned, an
    # "Unknown" category for this column -- the encoder would not know what
    # to do with one.


class DeliveryOrder(BaseModel):
    """A single raw order, in the same shape as the source dataset's columns
    (minus Order_ID and the target, Delivery_Time_min, which is what this
    endpoint predicts). Field names intentionally match the dataset's
    original column names used throughout this project's EDA and model
    documentation, so a request body can be read directly against those docs.
    """

    # use_enum_values=True makes the validated model store plain strings
    # ("Clear") for the Enum fields below instead of Enum objects
    # (Weather.CLEAR) -- simpler to hand off to the model_pipeline code,
    # which expects plain strings.
    model_config = ConfigDict(use_enum_values=True)

    Distance_km: float = Field(..., gt=0, description="Delivery distance in kilometers.")
    Preparation_Time_min: float = Field(..., gt=0, description="Food preparation time in minutes.")
    Weather: Weather
    Traffic_Level: TrafficLevel
    Time_of_Day: TimeOfDay
    Vehicle_Type: VehicleType
    Courier_Experience_yrs: Optional[float] = Field(
        default=None,
        ge=0,
        description=(
            "Courier experience in years. Optional: if omitted, the trained "
            "pipeline imputes it with the median learned from the training "
            "data (model_pipeline/data_preprocessing.py), so a caller "
            "without this information does not need to guess a value."
        ),
    )


class PredictionResponse(BaseModel):
    predicted_delivery_time_min: float

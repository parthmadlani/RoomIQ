# Data dictionary

Raw: 10,129 rows x 19 columns. 16 sensor-related fields, two time fields and one target. CO2 slope is source-derived.

| column | role | raw_dtype | processed_dtype | units | missing | unique_values | min | max | example | meaning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Date | metadata | object | object | YYYY/MM/DD | 0 | 7 | 2017/12/22 | 2018/01/11 | 2017/12/22 | Recorded date; metadata only |
| Time | metadata | object | object | HH:MM:SS | 0 | 10129 | 00:00:28 | 23:59:58 | 10:49:41 | Recorded time; timezone not supplied |
| S1_Temp | sensor_related | float64 | float64 | degrees C | 0 | 24 | 24.94 | 26.38 | 24.94 | Temperature at node S1 |
| S2_Temp | sensor_related | float64 | float64 | degrees C | 0 | 69 | 24.75 | 29 | 24.75 | Temperature at node S2 |
| S3_Temp | sensor_related | float64 | float64 | degrees C | 0 | 29 | 24.44 | 26.19 | 24.56 | Temperature at node S3 |
| S4_Temp | sensor_related | float64 | float64 | degrees C | 0 | 27 | 24.94 | 26.56 | 25.38 | Temperature at node S4 |
| S1_Light | sensor_related | int64 | int64 | lux | 0 | 68 | 0 | 165 | 121 | Light at node S1 |
| S2_Light | sensor_related | int64 | int64 | lux | 0 | 82 | 0 | 258 | 34 | Light at node S2 |
| S3_Light | sensor_related | int64 | int64 | lux | 0 | 177 | 0 | 280 | 53 | Light at node S3 |
| S4_Light | sensor_related | int64 | int64 | lux | 0 | 75 | 0 | 74 | 40 | Light at node S4 |
| S1_Sound | sensor_related | float64 | float64 | V | 0 | 231 | 0.06 | 3.88 | 0.08 | Sound amplifier voltage at node S1, not decibels |
| S2_Sound | sensor_related | float64 | float64 | V | 0 | 185 | 0.04 | 3.44 | 0.19 | Sound amplifier voltage at node S2, not decibels |
| S3_Sound | sensor_related | float64 | float64 | V | 0 | 258 | 0.04 | 3.67 | 0.06 | Sound amplifier voltage at node S3, not decibels |
| S4_Sound | sensor_related | float64 | float64 | V | 0 | 106 | 0.05 | 3.4 | 0.06 | Sound amplifier voltage at node S4, not decibels |
| S5_CO2 | sensor_related | int64 | int64 | ppm | 0 | 186 | 345 | 1270 | 390 | CO2 concentration at node S5 |
| S5_CO2_Slope | sensor_related | float64 | float64 | unspecified | 0 | 1579 | -6.296 | 8.981 | 0.7692 | Supplied CO2 window slope; alignment and time units unverified |
| S6_PIR | sensor_related | int64 | int64 | 0/1 | 0 | 2 | 0 | 1 | 0 | Motion detection at node S6; not a count of people |
| S7_PIR | sensor_related | int64 | int64 | 0/1 | 0 | 2 | 0 | 1 | 0 | Motion detection at node S7; not a count of people |
| Room_Occupancy_Count | target | int64 | int64 | people | 0 | 4 | 0 | 3 | 1 | Manually recorded occupancy class, 0 to 3 |

## Added fields

| field | dtype | definition | purpose |
| --- | --- | --- | --- |
| Timestamp | datetime64[ns] | Date + Time, strict parsing; timezone unknown | Audit and future splits |
| Source_Row | int64 | Original CSV line number starting at 2 | Traceability |
| Recording_Segment | int64 | New group after gap > 600 seconds | Descriptive grouping, not experimental label |
| Avg_Temp | float64 | (S1_Temp + S2_Temp + S3_Temp + S4_Temp) / 4 | Optional compact feature, degrees C |
| Avg_Light | float64 | (S1_Light + S2_Light + S3_Light + S4_Light) / 4 | Optional compact feature, lux |
| Avg_Sound | float64 | (S1_Sound + S2_Sound + S3_Sound + S4_Sound) / 4 | Optional compact feature, V |
| PIR_Any | int64 | max(S6_PIR, S7_PIR) | Logical OR, not a people count |

CSV does not store dtype metadata. Load Timestamp with parse_dates. Date/Time remain strings; numeric originals keep their values and units. Units follow UCI metadata; the slope time units are unspecified.

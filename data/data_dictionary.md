# Uber Eats Analytics System - Data Dictionary

### 1. customers.csv
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `customer_id` | String (PK) | Unique customer identifier |
| `name` | String | Customer full name |
| `email` | String | Contact email address |
| `age` | Integer | Customer age (18 - 70) |
| `customer_tier` | Categorical | Loyalty tier (Bronze, Silver, Gold, UberOne) |
| `city_zone` | Categorical | Customer residential urban sector |
| `signup_date` | Date | Account creation date |

### 2. restaurants.csv
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `restaurant_id` | String (PK) | Unique restaurant identifier |
| `name` | String | Restaurant name |
| `category` | Categorical | Cuisine category |
| `city_zone` | Categorical | Geographic kitchen location |
| `average_rating` | Float | Historical food rating (1.0 - 5.0) |
| `price_range` | Categorical | Pricing category ($, $$, $$$) |
| `base_prep_time_min` | Integer | Baseline cooking time in minutes |

### 3. drivers.csv
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `driver_id` | String (PK) | Unique courier identifier |
| `name` | String | Driver full name |
| `vehicle_type` | Categorical | Bicycle, Scooter, Motorcycle, or Car |
| `rating` | Float | Average courier rating |
| `total_completed_trips` | Integer | Lifetime completed trip count |
| `active_status` | Categorical | Active or Inactive status |

### 4. orders.csv
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `order_id` | String (PK) | Unique order identifier |
| `customer_id` | String (FK) | Reference to customers table |
| `restaurant_id` | String (FK) | Reference to restaurants table |
| `order_timestamp` | Datetime | Order placement timestamp (YYYY-MM-DD HH:MM:SS) |
| `num_items` | Integer | Number of items in order basket |
| `order_amount` | Float | Total dollar value of the food ordered |
| `order_status` | Categorical | Delivered, Cancelled, or Refunded |

### 5. deliveries.csv
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `delivery_id` | String (PK) | Unique delivery identifier |
| `order_id` | String (FK) | Reference to orders table |
| `driver_id` | String (FK) | Reference to drivers table |
| `delivery_distance_km` | Float | Travel distance from restaurant to customer |
| `prep_time_minutes` | Integer | Measured kitchen preparation duration |
| `delivery_time_minutes` | Float | Total transit + prep duration (Target for Regression) |
| `weather` | Categorical | Clear, Rainy, Cloudy, Snowy, Windy |
| `traffic_condition` | Categorical | Low, Medium, High, Congested |
| `tip_amount` | Float | Gratuity added by customer (Target for Regression) |

### 6. reviews.csv
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `review_id` | String (PK) | Unique review identifier |
| `order_id` | String (FK) | Reference to orders table |
| `customer_id` | String (FK) | Reference to customers table |
| `rating` | Integer | Star rating (1 - 5) |
| `review_text` | Text | Unstructured customer feedback |
| `sentiment_label` | Categorical | Ground truth sentiment (Positive, Neutral, Negative) |

### 7. payments.csv
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `payment_id` | String (PK) | Unique transaction identifier |
| `order_id` | String (FK) | Reference to orders table |
| `payment_method` | Categorical | Credit Card, Debit Card, Apple Pay, PayPal, etc. |
| `payment_status` | Categorical | Successful or Refunded |
| `transaction_timestamp`| Datetime | Timestamp of processed transaction |
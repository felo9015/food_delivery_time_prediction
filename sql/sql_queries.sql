-- =============================================================================
-- Part I: SQL — Final Queries
--
-- These are the clean, validated versions of the 5 required queries. Each was
-- built and tested step by step in sql/sql_exploration.ipynb against a
-- synthetic dataset matching the schema below; that notebook documents the
-- reasoning, the intermediate steps, and the edge cases used to validate each
-- query. Assumed schema (as given in the assessment prompt):
--
-- deliveries(delivery_id, delivery_person_id, restaurant_area, customer_area,
--            delivery_distance_km, delivery_time_min, order_placed_at,
--            weather_condition, traffic_condition, delivery_rating)
-- delivery_persons(delivery_person_id, name, region, hired_date, is_active)
-- restaurants(restaurant_id, area, name, cuisine_type, avg_preparation_time_min)
-- orders(order_id, delivery_id, restaurant_id, customer_id, order_value, items_count)
--
-- Note: deliveries.delivery_person_id is VARCHAR while
-- delivery_persons.delivery_person_id is INT, per the original schema — a
-- CAST is used wherever the two tables are joined.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- Question 1: Top 5 customer areas with the highest average delivery time
-- in the last 30 days.
-- -----------------------------------------------------------------------------
SELECT
    customer_area,
    AVG(delivery_time_min) AS avg_delivery_time_min
FROM deliveries
WHERE order_placed_at >= CURRENT_DATE - INTERVAL 30 DAY
GROUP BY customer_area
ORDER BY avg_delivery_time_min DESC
LIMIT 5;


-- -----------------------------------------------------------------------------
-- Question 2: Average delivery time per traffic condition, by restaurant
-- area and cuisine type.
--
-- Joined through orders.restaurant_id (not deliveries.restaurant_area =
-- restaurants.area) to identify the exact restaurant behind each delivery.
-- Joining on the area name alone would match every restaurant located in
-- that area, duplicating each delivery once per restaurant sharing the area
-- and silently inflating the averages.
-- -----------------------------------------------------------------------------
SELECT
    d.traffic_condition,
    r.area AS restaurant_area,
    r.cuisine_type,
    AVG(d.delivery_time_min) AS avg_delivery_time_min,
    COUNT(*) AS n_deliveries
FROM deliveries d
JOIN orders o ON d.delivery_id = o.delivery_id
JOIN restaurants r ON o.restaurant_id = r.restaurant_id
GROUP BY d.traffic_condition, r.area, r.cuisine_type
ORDER BY r.area, r.cuisine_type, d.traffic_condition;


-- -----------------------------------------------------------------------------
-- Question 3: Top 10 delivery people with the fastest average delivery time,
-- considering only those with at least 50 deliveries and who are still
-- active.
-- -----------------------------------------------------------------------------
SELECT
    dp.delivery_person_id,
    dp.name,
    AVG(d.delivery_time_min) AS avg_delivery_time_min,
    COUNT(*) AS n_deliveries
FROM deliveries d
JOIN delivery_persons dp ON d.delivery_person_id = CAST(dp.delivery_person_id AS VARCHAR)
WHERE dp.is_active = TRUE
GROUP BY dp.delivery_person_id, dp.name
HAVING COUNT(*) >= 50
ORDER BY avg_delivery_time_min ASC
LIMIT 10;


-- -----------------------------------------------------------------------------
-- Question 4: The most profitable restaurant area in the last 3 months,
-- defined as the area with the highest total order value.
-- -----------------------------------------------------------------------------
SELECT
    d.restaurant_area,
    SUM(o.order_value) AS total_order_value
FROM orders o
JOIN deliveries d ON o.delivery_id = d.delivery_id
WHERE d.order_placed_at >= CURRENT_DATE - INTERVAL 3 MONTH
GROUP BY d.restaurant_area
ORDER BY total_order_value DESC
LIMIT 1;


-- -----------------------------------------------------------------------------
-- Question 5: Identify whether any delivery people show an increasing trend
-- in average delivery time.
--
-- Trend detection is not a single SQL query: it requires fitting a linear
-- regression (delivery_time_min as a function of days since each courier's
-- first delivery) per courier, then testing statistical significance. SQL
-- extracts the base, per-delivery data below; the regression, the dynamic
-- minimum-deliveries threshold, and the significance test are completed in
-- Python (scipy.stats.linregress) in sql/sql_exploration.ipynb, in the
-- "Question 5" section. See that notebook for the full method and results.
-- -----------------------------------------------------------------------------
SELECT
    delivery_person_id,
    order_placed_at,
    delivery_time_min
FROM deliveries;

-- bqml/02_top_skus.sql
-- Materializes the top 200 SKUs by 2024 revenue.
-- Used to filter BQML training to a tractable size for the lab.

CREATE OR REPLACE TABLE ml.top_skus AS
SELECT
  sku_id,
  SUM(net_revenue) AS rev_2024,
  RANK() OVER (ORDER BY SUM(net_revenue) DESC) AS revenue_rank
FROM curated.fact_sales_daily
WHERE sale_year = 2024
GROUP BY sku_id
QUALIFY revenue_rank <= 200
ORDER BY revenue_rank;
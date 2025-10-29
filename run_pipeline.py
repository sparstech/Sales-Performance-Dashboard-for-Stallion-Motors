
#!/usr/bin/env python3
\"\"\"run_pipeline.py

Automatic monthly pipeline script:
- regenerates monthly aggregates (stallion_monthly_agg.csv)
- computes salesperson leaderboard and saves CSV
- attempts to run Prophet (recommended) to create 6-month forecast CSV ('stallion_prophet_forecast.csv')
- if Prophet not available, tries pmdarima ARIMA fallback to produce 'stallion_arima_forecast.csv'

Usage:
    python run_pipeline.py --sales_csv stallion_sales_data.csv

This script is safe to run on a schedule (cron, GitHub Actions, etc.).
\"\"\"
import argparse, sys
import pandas as pd
from pathlib import Path

PROJ = Path(__file__).resolve().parent

def regenerate_monthly_agg(df):
    monthly = df.groupby(pd.Grouper(key='Date', freq='M')).agg(\n        Revenue=('Price','sum'),\n        Units=('Quantity','sum'),\n        Profit=('Profit','sum')\n    ).reset_index()\n    monthly = monthly.rename(columns={'Date':'ds'})\n    monthly.to_csv(PROJ / 'stallion_monthly_agg.csv', index=False)\n    print('Saved stallion_monthly_agg.csv')\n    return monthly\n\ndef create_leaderboard(df):\n    # Lazy import to avoid dependency if user only wants basic pipeline\n    from commission_calc import compute_leaderboard\n    leaderboard = compute_leaderboard(df)\n    leaderboard.to_csv(PROJ / 'salesperson_leaderboard_monthly.csv', index=False)\n    print('Saved salesperson_leaderboard_monthly.csv')\n    return leaderboard\n\n\ndef try_prophet_forecast(monthly_df, periods=6):\n    try:\n        from prophet import Prophet\n        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)\n        dfp = monthly_df[['ds','Revenue']].rename(columns={'ds':'ds','Revenue':'y'})\n        dfp['ds'] = pd.to_datetime(dfp['ds'])\n        m.fit(dfp)\n        future = m.make_future_dataframe(periods=periods, freq='M')\n        forecast = m.predict(future)\n        forecast[['ds','yhat','yhat_lower','yhat_upper']].to_csv(PROJ / 'stallion_prophet_forecast.csv', index=False)\n        print('Saved stallion_prophet_forecast.csv (Prophet)')\n        return True\n    except Exception as e:\n        print('Prophet forecasting failed or not installed:', e)\n        return False\n\n\ndef try_arima_forecast(monthly_df, periods=6):\n    try:\n        import pmdarima as pm\n        series = monthly_df.set_index('ds')['Revenue']\n        series.index = pd.to_datetime(series.index)\n        model = pm.auto_arima(series, seasonal=True, m=12, error_action='ignore', suppress_warnings=True)\n        fc = model.predict(n_periods=periods)\n        last_date = series.index.max()\n        future_index = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=periods, freq='M')\n        arima_df = pd.DataFrame({'ds': future_index, 'yhat': fc})\n        arima_df.to_csv(PROJ / 'stallion_arima_forecast.csv', index=False)\n        print('Saved stallion_arima_forecast.csv (ARIMA)')\n        return True\n    except Exception as e:\n        print('ARIMA forecasting failed or not installed:', e)\n        return False\n\n\ndef main(sales_csv):\n    df = pd.read_csv(sales_csv, parse_dates=['Date'])\n    # regenerate monthly aggregates\n    monthly = regenerate_monthly_agg(df)\n    # create leaderboard\n    create_leaderboard(df)\n    # try prophet\n    ok = try_prophet_forecast(monthly)\n    if not ok:\n        # fallback to ARIMA\n        try_arima_forecast(monthly)\n\nif __name__ == '__main__':\n    parser = argparse.ArgumentParser()\n    parser.add_argument('--sales_csv', default=str(PROJ / 'stallion_sales_data.csv'))\n    args = parser.parse_args()\n    main(args.sales_csv)\n
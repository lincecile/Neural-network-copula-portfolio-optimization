import pandas as pd
from Clean_df_paper import df_total_set
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.figure(figsize=(10, 6))
plt.plot(df_total_set.index, df_total_set[df_total_set.columns[0]], label=df_total_set.columns[0])
plt.plot(df_total_set.index, df_total_set[df_total_set.columns[1]], label=df_total_set.columns[1])
plt.plot(df_total_set.index, df_total_set[df_total_set.columns[2]], label=df_total_set.columns[2])

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('ETF CLOSING PRICE')
plt.legend()
plt.grid(True)

plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

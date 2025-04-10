'''
Today is the 20250404, I want to test the new platform for tip_onet2025 with cursor and remote configuration.
'''

#1. get the local time in retomte server
from datetime import datetime
import pytz

shanghai_tz = pytz.timezone('Asia/Shanghai')

# Get current time in Shanghai
shanghai_time = datetime.now(shanghai_tz)

# Format the time as string
shanghai_time_str = shanghai_time.strftime("%Y-%m-%d %H:%M:%S %Z")
# Format time to show only year, month, day and hour
shanghai_time_formatted = shanghai_time.strftime("%Y%m%d_%H")
print(f"Current Shanghai time (YYYYMMDD_HH): {shanghai_time_formatted}")





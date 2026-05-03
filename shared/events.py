"""Shared financial event helpers used by both trading and study pages."""
import datetime


def _nth_weekday(year, month, weekday, n):
    """Return the nth occurrence of a weekday in a given month (1-indexed).
    weekday: 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri"""
    first = datetime.date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + datetime.timedelta(days=offset + 7 * (n - 1))


# Hand-curated FOMC meeting dates (final day of each meeting).
# Sources: Federal Reserve historical calendars + 2025-2026 published schedule.
FOMC_DATES = [
    # 2014
    datetime.date(2014, 1, 29), datetime.date(2014, 3, 19),
    datetime.date(2014, 4, 30), datetime.date(2014, 6, 18),
    datetime.date(2014, 7, 30), datetime.date(2014, 9, 17),
    datetime.date(2014, 10, 29), datetime.date(2014, 12, 17),
    # 2015
    datetime.date(2015, 1, 28), datetime.date(2015, 3, 18),
    datetime.date(2015, 4, 29), datetime.date(2015, 6, 17),
    datetime.date(2015, 7, 29), datetime.date(2015, 9, 17),
    datetime.date(2015, 10, 28), datetime.date(2015, 12, 16),
    # 2016
    datetime.date(2016, 1, 27), datetime.date(2016, 3, 16),
    datetime.date(2016, 4, 27), datetime.date(2016, 6, 15),
    datetime.date(2016, 7, 27), datetime.date(2016, 9, 21),
    datetime.date(2016, 11, 2), datetime.date(2016, 12, 14),
    # 2017
    datetime.date(2017, 2, 1), datetime.date(2017, 3, 15),
    datetime.date(2017, 5, 3), datetime.date(2017, 6, 14),
    datetime.date(2017, 7, 26), datetime.date(2017, 9, 20),
    datetime.date(2017, 11, 1), datetime.date(2017, 12, 13),
    # 2018
    datetime.date(2018, 1, 31), datetime.date(2018, 3, 21),
    datetime.date(2018, 5, 2), datetime.date(2018, 6, 13),
    datetime.date(2018, 8, 1), datetime.date(2018, 9, 26),
    datetime.date(2018, 11, 8), datetime.date(2018, 12, 19),
    # 2019
    datetime.date(2019, 1, 30), datetime.date(2019, 3, 20),
    datetime.date(2019, 5, 1), datetime.date(2019, 6, 19),
    datetime.date(2019, 7, 31), datetime.date(2019, 9, 18),
    datetime.date(2019, 10, 30), datetime.date(2019, 12, 11),
    # 2020
    datetime.date(2020, 1, 29), datetime.date(2020, 3, 3),  # emergency cut
    datetime.date(2020, 3, 15),  # emergency cut
    datetime.date(2020, 3, 18), datetime.date(2020, 4, 29),
    datetime.date(2020, 6, 10), datetime.date(2020, 7, 29),
    datetime.date(2020, 9, 16), datetime.date(2020, 11, 5),
    datetime.date(2020, 12, 16),
    # 2021
    datetime.date(2021, 1, 27), datetime.date(2021, 3, 17),
    datetime.date(2021, 4, 28), datetime.date(2021, 6, 16),
    datetime.date(2021, 7, 28), datetime.date(2021, 9, 22),
    datetime.date(2021, 11, 3), datetime.date(2021, 12, 15),
    # 2022
    datetime.date(2022, 1, 26), datetime.date(2022, 3, 16),
    datetime.date(2022, 5, 4), datetime.date(2022, 6, 15),
    datetime.date(2022, 7, 27), datetime.date(2022, 9, 21),
    datetime.date(2022, 11, 2), datetime.date(2022, 12, 14),
    # 2023
    datetime.date(2023, 2, 1), datetime.date(2023, 3, 22),
    datetime.date(2023, 5, 3), datetime.date(2023, 6, 14),
    datetime.date(2023, 7, 26), datetime.date(2023, 9, 20),
    datetime.date(2023, 11, 1), datetime.date(2023, 12, 13),
    # 2024
    datetime.date(2024, 1, 31), datetime.date(2024, 3, 20),
    datetime.date(2024, 5, 1), datetime.date(2024, 6, 12),
    datetime.date(2024, 7, 31), datetime.date(2024, 9, 18),
    datetime.date(2024, 11, 7), datetime.date(2024, 12, 18),
    # 2025
    datetime.date(2025, 1, 29), datetime.date(2025, 3, 19),
    datetime.date(2025, 5, 7),  datetime.date(2025, 6, 18),
    datetime.date(2025, 7, 30), datetime.date(2025, 9, 17),
    datetime.date(2025, 10, 29), datetime.date(2025, 12, 17),
    # 2026
    datetime.date(2026, 1, 28), datetime.date(2026, 3, 18),
    datetime.date(2026, 5, 6),  datetime.date(2026, 6, 17),
    datetime.date(2026, 7, 29), datetime.date(2026, 9, 16),
    datetime.date(2026, 10, 28), datetime.date(2026, 12, 16),
]


def get_financial_events(start_date, end_date):
    """Return sorted list of (date, label) for financial events in range."""
    start_d = start_date.date() if hasattr(start_date, 'date') else start_date
    end_d = end_date.date() if hasattr(end_date, 'date') else end_date
    events = []

    current = datetime.date(start_d.year, start_d.month, 1)
    while current <= end_d:
        y, m = current.year, current.month

        opex = _nth_weekday(y, m, 4, 3)
        if start_d <= opex <= end_d:
            events.append((opex, f"{opex.strftime('%b')} OPEX"))

        next_y, next_m = (y, m + 1) if m < 12 else (y + 1, 1)
        vix_exp = _nth_weekday(next_y, next_m, 4, 3) - datetime.timedelta(days=30)
        if start_d <= vix_exp <= end_d:
            events.append((vix_exp, "VIX Exp"))

        if m == 11:
            tday = _nth_weekday(y, 11, 3, 4)
            if start_d <= tday <= end_d:
                events.append((tday, "Thanksgiving"))

        if m == 12:
            xmas = datetime.date(y, 12, 25)
            nye = datetime.date(y, 12, 31)
            if start_d <= xmas <= end_d:
                events.append((xmas, "Xmas"))
            if start_d <= nye <= end_d:
                events.append((nye, "NYE"))

        current = datetime.date(y + (1 if m == 12 else 0), (m % 12) + 1, 1)

    for d in FOMC_DATES:
        if start_d <= d <= end_d:
            events.append((d, "FOMC"))

    events.sort(key=lambda x: x[0])
    return events

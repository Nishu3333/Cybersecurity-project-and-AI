# load_static_indicators.py

occupation_risk = {
    "ANY ACCOUNTANT": "High",
    "ANY ARTIST (ART/CRAFTPERSONS)": "High",
    "ANY ATTORNEY/ LAWYER": "Medium",
    "GOVT. OFFICIAL (OFFICER TO BELOW SECRETARY )": "High",
    "GOVT. JUDGE": "High",
    "SELF EMPLOYED JEWELERS": "High",
    "REAL ESTATE BUSINESS": "High",
    "CASINOS": "High",
    "GAMING -CASINOS": "High",
    "ENGINEER": "Low",
    "DOCTOR": "Low",
    "TEACHER": "Low",
    "FARMER": "Low",
    "STUDENT": "Low",
    "RETIRED": "Low",
    "SELF EMPLOYED LEGAL PROFESS/CONSULTANTS": "High",
    "GOVT. ARMY (CAPTAIN AND ABOVE)": "High",
    "GOVT. LEGISLATOR": "High"
}

country_risk = {
    "AFGHANISTAN": "High",
    "PAKISTAN": "High",
    "IRAN": "High",
    "SYRIA": "High",
    "RUSSIAN FEDERATION": "High",
    "NORTH KOREA": "High",
    "MYANMAR": "High",
    "VENEZUELA": "High",
    "CAMBODIA": "High",
    "SUDAN": "High",
    "YEMEN": "High",
    "ZIMBABWE": "High",
    "INDIA": "Medium",
    "CHINA": "Medium",
    "USA": "Low",
    "EUROPEAN UNION": "Low",
    "NEPAL": "Medium"
}

account_opening_method_risk = {
    "SOCIAL MEDIA": "High",
    "ONLINE ACCOUNT": "High",
    "EVENTS AND EXHIBITION": "Medium",
    "WALKIN CUSTOMER": "Low",
    "REFERRAL": "Low",
    "OFFLINE": "Low",        # Replaced CKPU with 'Offline'
    "MAIL": "Low",
    "SWIFT": "Low"
}

identification_used_risk = {
    "CITIZENSHIP": "Low",
    "PASSPORT": "Low",
    "ADHAR CARD": "High",
    "NRN ID": "Medium"
}
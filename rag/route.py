import re

# по каким словам в вопросе узнаём организацию
ORG_PATTERNS = {
    "sber": r"сбербанк|сбер\b|сбера\b",
    "vtb": r"втб",
    "tbank": r"т[\s\-‑–—]?банк|тинькоф|ткс",
    "domrf": r"дом\.?\s?рф",
    "mkb": r"мкб|московск\w+ кредитн\w+ банк",
    "rshb": r"рсхб|россельхоз|сельскохозяйственн\w+ банк",
    "sovkom": r"совком",
    "alfa": r"альфа",
    "gpb": r"газпромбанк|гпб|нпф",
}
HALF_YEAR = r"шест\w+ месяц|6 месяц|полугоди|30 июня|промежуточн|1 июля"


def find_orgs(question):
    text = question.lower()
    return [org for org, pattern in ORG_PATTERNS.items() if re.search(pattern, text)]


def find_years(question):
    return sorted({int(y) for y in re.findall(r"\b(20[12]\d)\b", question)})


def rank_documents(question, doc_ids):
    # возвращает подходящие отчёты и вес каждого: чем больше, тем вероятнее ответ в нём
    orgs = find_orgs(question)
    years = find_years(question)
    half = bool(re.search(HALF_YEAR, question.lower()))
    weights = {}
    for doc_id in doc_ids:
        org, year, period = doc_id.rsplit("_", 2)
        year = int(year)
        if orgs and org not in orgs:
            continue
        weight = 0.0
        if years:
            if year == max(years):
                weight = 1.0
            elif year in years or year == max(years) + 1:
                # цифры за прошлый год есть и в следующем отчёте как сравнительные
                weight = 0.5
            if weight and (period == "6m") != half:
                weight -= 0.25
        weights[doc_id] = weight
    return weights

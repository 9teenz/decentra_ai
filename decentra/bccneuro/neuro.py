# Запуск:
#   python neuro.py --raw raw --clients users/clients.csv --out output

from __future__ import annotations

import os
import re
import json
import random
import argparse
from datetime import datetime, timedelta
from glob import glob
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score

# --------------------------- Конфигурация ---------------------------
WINDOW_DAYS = 90
TODAY = datetime.now().date()
START_DATE = TODAY - timedelta(days=WINDOW_DAYS)

# Список продуктов
PRODUCTS = [
    "Карта для путешествий",
    "Премиальная карта",
    "Кредитная карта",
    "Обмен валют",
    "Кредит наличными",
    "Депозит Мультивалютный",
    "Депозит Сберегательный",
    "Депозит Накопительный",
    "Инвестиции",
    "Золотые слитки"
]


CATEGORIES = [
    "Одежда и обувь", "Продукты питания", "Кафе и рестораны", "Медицина", "Авто", "Спорт",
    "Развлечения", "АЗС", "Кино", "Питомцы", "Книги", "Цветы", "Едим дома", "Смотрим дома",
    "Играем дома", "Косметика и Парфюмерия", "Подарки", "Ремонт дома", "Мебель", "Спа и массаж",
    "Ювелирные украшения", "Такси", "Отели", "Путешествия", "Онлайн-покупки", "Страхование",
    "Обмен валют", "Международные переводы"
]

TRANSFER_TYPES = [
    "salary_in", "stipend_in", "family_in", "cashback_in", "refund_in", "card_in",
    "p2p_out", "card_out", "atm_withdrawal", "utilities_out", "loan_payment_out",
    "cc_repayment_out", "installment_payment_out", "fx_buy", "fx_sell",
    "invest_out", "invest_in", "deposit_topup_out", "deposit_fx_topup_out",
    "deposit_fx_withdraw_in", "gold_buy_out", "gold_sell_in", "swift_out", "swift_in"
]

NUM_CLIENTS = 60

random.seed(42)
np.random.seed(42)

# --------------------------- Утилиты ---------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding='cp1251')
        except Exception:
            return pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def format_amount(amount: float | int, currency_symbol: str = '₸') -> str:
    if amount is None or (isinstance(amount, float) and np.isnan(amount)):
        return f"0 {currency_symbol}"
    try:
        return f"{int(round(amount)):,}".replace(",", " ") + f" {currency_symbol}"
    except Exception:
        return f"{amount} {currency_symbol}"

def json_friendly(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (list, tuple, set)):
        return [json_friendly(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): json_friendly(v) for k, v in obj.items()}
    try:
        return str(obj)
    except Exception:
        return None

def find_client_code_in_filename(fname: str) -> int | None:
    base = os.path.basename(fname)
    m = re.search(r"(\d{1,6})", base)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

# --------------------------- Генерация синтетики ---------------------------

def generate_synthetic(raw_dir: str, clients_csv: str, n_clients: int = NUM_CLIENTS) -> None:
    ensure_dir(raw_dir)
    first_names = [
        "Альберт","Алина","Бакыт","Виктор","Гульнара","Данияр","Ержан","Жания","Зухра","Игорь",
        "Камила","Ляззат","Марат","Нурсултан","Ольга","Павел","Рамазан","Салтанат","Тимур","Улжан",
        "Фарух","Ханна","Шолпан","Эрнест","Юлия","Ярослав","Алия","Дания","Айбек","Сабина"
    ]

    clients = []
    for i in range(1, n_clients + 1):
        name = random.choice(first_names) + ("" if random.random() < 0.6 else " " + random.choice(["И.","М.","А."]))
        status = random.choices(["Студент", "Зарплатный клиент", "Премиальный клиент", "Стандартный клиент"], weights=[0.15,0.35,0.15,0.35])[0]
        age = random.randint(18, 70)
        city = random.choice(["Нур-Султан","Алматы","Шымкент","Караганда","Костанай","Актобе"])
        if status == "Премиальный клиент":
            avg_balance = random.randint(1_500_000, 8_000_000)
        elif status == "Зарплатный клиент":
            avg_balance = random.randint(80_000, 800_000)
        elif status == "Студент":
            avg_balance = random.randint(10_000, 80_000)
        else:
            avg_balance = random.randint(30_000, 400_000)
        clients.append({
            "client_code": i,
            "name": name,
            "status": status,
            "age": age,
            "city": city,
            "avg_monthly_balance_KZT": avg_balance
        })

    df_clients = pd.DataFrame(clients)
    df_clients.to_csv(clients_csv, index=False, encoding='utf-8')
    print("Saved clients ->", clients_csv)

    
    all_tx = []
    all_xf = []
    for c in clients:
        num_tx = random.randint(120, 320)
        weights = []
        for cat in CATEGORIES:
            base = 1.0
            if c["status"] == "Премиальный клиент":
                if cat in ["Отели","Путешествия","АЗС","Ювелирные украшения","Кафе и рестораны","Международные переводы"]:
                    base = 2.5
                elif cat in ["Продукты питания","Едим дома"]:
                    base = 1.2
            elif c["status"] == "Студент":
                if cat in ["Кино","Играем дома","Смотрим дома","Онлайн-покупки","Кафе и рестораны"]:
                    base = 2.0
                if cat == "Ювелирные украшения":
                    base = 0.2
            elif c["status"] == "Зарплатный клиент":
                if cat in ["АЗС","Путешествия","Кафе и рестораны","Продукты питания","Онлайн-покупки"]:
                    base = 1.5
            else:
                if cat in ["Продукты питания","АЗС","Мебель","Ремонт дома"]:
                    base = 1.3
            weights.append(base)

        for _ in range(num_tx):
            d = START_DATE + timedelta(days=random.randint(0, WINDOW_DAYS - 1))
            cat = random.choices(CATEGORIES, weights=weights)[0]
            
            if cat in ["Ювелирные украшения","Мебель"]:
                amount = int(max(10_000, random.gauss(150_000, 120_000)))
            elif cat in ["Путешествия","Отели","Международные переводы"]:
                amount = int(max(5_000, random.gauss(60_000, 50_000)))
            elif cat in ["АЗС","Такси"]:
                amount = int(max(200, random.gauss(6_000, 4_000)))
            elif cat in ["Кафе и рестораны","Кино","Косметика и Парфюмерия","Развлечения"]:
                amount = int(max(500, random.gauss(4_000, 3_000)))
            elif cat in ["Продукты питания","Едим дома"]:
                amount = int(max(200, random.gauss(12_000, 8_000)))
            elif cat == "Обмен валют":
                
                amount = int(max(500, random.gauss(40_000, 30_000)))
            elif cat == "Онлайн-покупки":
                amount = int(max(200, random.gauss(8_000, 12_000)))
            else:
                amount = int(max(200, random.gauss(8_000, 10_000)))

            all_tx.append({
                "date": d.isoformat(),
                "category": cat,
                "amount": abs(amount),
                "currency": random.choice(["KZT","KZT","KZT","USD","EUR"]),
                "client_code": c["client_code"]
            })

        # переводы
        num_xf = random.randint(40, 120)
        for _ in range(num_xf):
            d = START_DATE + timedelta(days=random.randint(0, WINDOW_DAYS - 1))
            typ = random.choice(TRANSFER_TYPES + ["swift_out","swift_in"])
            direction = "in" if typ.endswith("_in") or typ in ["salary_in","stipend_in","family_in","card_in","cashback_in","refund_in","gold_sell_in","invest_in","swift_in"] else random.choice(["in","out"])
            if typ == "salary_in":
                amount = random.randint(80_000, 400_000)
            elif typ == "stipend_in":
                amount = random.randint(10_000, 60_000)
            elif typ in ["p2p_out","card_out"]:
                amount = random.randint(200, 200_000)
            elif typ == "atm_withdrawal":
                amount = random.randint(1_000, 200_000)
            elif typ in ["fx_buy","fx_sell","deposit_fx_topup_out","deposit_fx_withdraw_in","gold_buy_out","gold_sell_in","swift_out","swift_in"]:
                amount = random.randint(1_000, 200_000)
            elif typ in ["loan_payment_out","cc_repayment_out","installment_payment_out"]:
                amount = random.randint(10_000, 300_000)
            else:
                amount = random.randint(200, 100_000)

            all_xf.append({
                "date": d.isoformat(),
                "type": typ,
                "direction": direction,
                "amount": abs(amount),
                "currency": random.choice(["KZT","USD","EUR","RUB"]),
                "client_code": c["client_code"]
            })

    tx_all_path = os.path.join(raw_dir, "transactions_all.csv")
    xf_all_path = os.path.join(raw_dir, "transfers_all.csv")
    pd.DataFrame(all_tx).to_csv(tx_all_path, index=False, encoding='utf-8')
    pd.DataFrame(all_xf).to_csv(xf_all_path, index=False, encoding='utf-8')
    print("Saved transactions_all ->", tx_all_path)
    print("Saved transfers_all ->", xf_all_path)

# --------------------------- Эвристика ---------------------------

def evaluate_benefits_for_client(client_row: pd.Series,
                                 transactions: pd.DataFrame,
                                 transfers: pd.DataFrame) -> Tuple[Dict[str,float], Dict[str,Any]]:
    """
    Полная эвристика по всем категориям.
    Цель: выдать релевантный продукт (или топ продуктов) на основе истории.
    Логика:
     - кредитная карта даётся ТОЛЬКО если есть реальная потребность (need_credit=True),
       либо если топ-траты сильно говорят в её пользу;
     - тревел / премиум / обмен валют / депозиты / инвестиции рассчитываются отдельно.
    Изменено: усилена логика для "Карта для путешествий" — учитываем количество поездок, долю АЗС/отелей,
    и даём более заметный буст, чтобы клиенты с поездками/АЗС получали релевантную рекомендацию.
    """
    client_code = client_row["client_code"]
    cli_trans = transactions[transactions["client_code"] == client_code] if not transactions.empty else pd.DataFrame(columns=["date","category","amount","currency","client_code"])
    cli_xfer = transfers[transfers["client_code"] == client_code] if not transfers.empty else pd.DataFrame(columns=["date","type","direction","amount","currency","client_code"])

    # суммирование трат по категориям
    if not cli_trans.empty:
        spend_by_cat = cli_trans.groupby("category", as_index=False)["amount"].sum().set_index("category").to_dict().get("amount", {})
    else:
        spend_by_cat = {}

    total_spend_3m = float(cli_trans["amount"].sum()) if len(cli_trans) > 0 else 0.0
    avg_monthly_spend = total_spend_3m / 3.0 if total_spend_3m > 0 else 0.0
    avg_balance = float(client_row.get("avg_monthly_balance_KZT", 0.0) or 0.0)

    # топ категорийф
    top_cats_items = sorted(spend_by_cat.items(), key=lambda x: x[1], reverse=True)[:5]
    top_cat_names = [c for c, v in top_cats_items]
    top3_spend = sum(v for _, v in top_cats_items[:3])

    # путешествия
    travel_cats = {"Путешествия", "Отели", "Такси"}
    travel_spend = sum(spend_by_cat.get(c, 0.0) for c in travel_cats)
    travel_tx_count = int(cli_trans[cli_trans.get("category", "").isin(travel_cats)].shape[0]) if len(cli_trans) > 0 else 0
    travel_ratio = travel_spend / (total_spend_3m + 1e-9)

    # развлечения
    entertainment_cats = {"Кино", "Развлечения", "Кафе и рестораны"}

    # онлайн покупки
    online_spend = spend_by_cat.get("Онлайн-покупки", 0.0) + spend_by_cat.get("Играем дома", 0.0) + spend_by_cat.get("Смотрим дома", 0.0)

    # валюты
    non_kzt_trans = cli_trans[cli_trans.get("currency","KZT") != "KZT"] if len(cli_trans) > 0 else pd.DataFrame()
    non_kzt_sum = float(non_kzt_trans["amount"].sum()) if len(non_kzt_trans) > 0 else 0.0
    fx_transfers = cli_xfer[cli_xfer.get("type","").isin(["fx_buy","fx_sell","deposit_fx_topup_out","deposit_fx_withdraw_in","swift_out","swift_in"])] if len(cli_xfer) > 0 else pd.DataFrame()
    fx_count = len(fx_transfers)
    fx_benefit_month = ((non_kzt_sum + (fx_transfers["amount"].sum() if len(fx_transfers) > 0 else 0.0)) * 0.005) / 3.0 + fx_count * 200.0

    # инвестиции
    invest_ops = cli_xfer[cli_xfer.get("type","").isin(["invest_in","invest_out"])] if len(cli_xfer) > 0 else pd.DataFrame()
    gold_ops = cli_xfer[cli_xfer.get("type","").isin(["gold_buy_out","gold_sell_in"])] if len(cli_xfer) > 0 else pd.DataFrame()


    
    large_spend = cli_trans[cli_trans["amount"] > 500_000] if "amount" in cli_trans.columns else pd.DataFrame()

    # нужен ли кредит
    need_credit = (len(large_spend) > 0) or (avg_balance < 100_000 and avg_monthly_spend > 80_000)

    benefits: Dict[str,float] = {}

    # ---------- Карта для путешествий ----------
    
    travel_cashback_month = (travel_spend * 0.07) / 3.0  
    travel_cashback_month += travel_tx_count * 250.0  
    
    azs_share = spend_by_cat.get("АЗС", 0.0) / (total_spend_3m + 1e-9)
    travel_cashback_month += (azs_share * travel_spend) * 0.01 / 3.0
    
    if travel_ratio > 0.05:
        travel_cashback_month += (travel_spend * 0.02) / 3.0
    
    if (non_kzt_sum > 0.0) and avg_balance > 200_000:
        travel_cashback_month += 500.0
    benefits["Карта для путешествий"] = max(0.0, min(travel_cashback_month, 200_000.0))

    # ---------- Премиальная карта ----------
    
    premium_special = sum(spend_by_cat.get(c, 0.0) for c in ["Ювелирные украшения", "Косметика и Парфюмерия", "Кафе и рестораны"])
    premium_score = 0.0
    if avg_balance >= 6_000_000:
        premium_base = 0.045
    elif avg_balance >= 1_000_000:
        premium_base = 0.03
    else:
        premium_base = 0.015
    premium_score = (total_spend_3m * premium_base) / 3.0 + (premium_special * 0.02) / 3.0
    
    if len(cli_xfer[cli_xfer.get("type","").isin(["swift_out","swift_in"])]) > 0:
        premium_score += 2000.0
    benefits["Премиальная карта"] = max(0.0, min(150_000.0, premium_score))

    # ---------- Кредитная карта ----------
    
    top3_val = top3_spend
    cc_cashback_month = 0.0
    if need_credit:
        
        cc_cashback_month = (top3_val * 0.05) / 3.0 + (online_spend * 0.03) / 3.0
    else:
        
        if top3_val > 500_000:
            cc_cashback_month = (top3_val * 0.02) / 3.0
        else:
            cc_cashback_month = 0.0
    benefits["Кредитная карта"] = max(0.0, cc_cashback_month)

    # ---------- Обмен валют ----------
    benefits["Обмен валют"] = max(0.0, float(fx_benefit_month))

    # ---------- Кредит наличными ----------
    benefits["Кредит наличными"] = 10_000.0 if need_credit else 0.0

    # ---------- Депозиты ----------
    deposit_multi_month = (avg_balance * 0.145) / 12.0 if avg_balance > 50_000 else 0.0
    deposit_sber_month = (avg_balance * 0.165) / 12.0 if avg_balance > 300_000 else 0.0
    deposit_nak_month = (avg_balance * 0.155) / 12.0 if avg_balance > 50_000 else 0.0
    benefits["Депозит Мультивалютный"] = max(0.0, deposit_multi_month)
    benefits["Депозит Сберегательный"] = max(0.0, deposit_sber_month)
    benefits["Депозит Накопительный"] = max(0.0, deposit_nak_month)

    # ---------- Инвестиции ----------
    invest_benefit_month = 0.0
    if len(invest_ops) > 0 or (avg_balance > 20_000 and avg_balance < 1_000_000):
        invest_benefit_month = (min(avg_balance, 500_000.0) * 0.10) / 12.0
    benefits["Инвестиции"] = max(0.0, invest_benefit_month)

    # ---------- Золото ----------
    gold_benefit_month = (min(avg_balance, 1_000_000.0) * 0.003) if (len(gold_ops) > 0 or avg_balance > 200_000) else 0.0
    benefits["Золотые слитки"] = max(0.0, gold_benefit_month)

    # ---------- усиления релевантности ----------
    
    if total_spend_3m > 0:
        share_products = spend_by_cat.get("Продукты питания", 0.0) / total_spend_3m
        share_ent = sum(spend_by_cat.get(c, 0.0) for c in entertainment_cats) / total_spend_3m if total_spend_3m > 0 else 0.0
        share_travel = travel_spend / total_spend_3m

        if share_products > 0.30:
            benefits["Депозит Накопительный"] += (spend_by_cat.get("Продукты питания", 0.0) * 0.003) / 3.0
            benefits["Премиальная карта"] += (spend_by_cat.get("Продукты питания", 0.0) * 0.002) / 3.0

        if share_ent > 0.18:
            benefits["Премиальная карта"] += (sum(spend_by_cat.get(c, 0.0) for c in entertainment_cats) * 0.02) / 3.0

        if share_travel > 0.06:
            benefits["Карта для путешествий"] += (travel_spend * 0.03) / 3.0
            if avg_balance > 500_000:
                benefits["Премиальная карта"] += (travel_spend * 0.01) / 3.0
        elif travel_ratio > 0.03:
            
            benefits["Карта для путешествий"] += (travel_spend * 0.01) / 3.0

        
        if online_spend / (total_spend_3m + 1e-9) > 0.2:
            
            benefits["Кредитная карта"] += (online_spend * 0.01) / 3.0
            benefits["Премиальная карта"] += (online_spend * 0.005) / 3.0

        
        if non_kzt_sum / (total_spend_3m + 1e-9) > 0.05 or fx_count > 0:
            benefits["Обмен валют"] += (non_kzt_sum * 0.02) / 3.0
            benefits["Депозит Мультивалютный"] += (non_kzt_sum * 0.01) / 3.0

    
    for k in list(benefits.keys()):
        if benefits[k] is None:
            benefits[k] = 0.0
        try:
            benefits[k] = float(benefits[k])
            if benefits[k] < 0:
                benefits[k] = 0.0
        except Exception:
            benefits[k] = 0.0

    # соберём контекст
    context = {
        "total_spend_3m": float(total_spend_3m),
        "avg_monthly_spend": float(avg_monthly_spend),
        "top_cats": list(top_cat_names),
        "travel_spend_3m": float(travel_spend),
        "avg_balance": float(avg_balance),
        "non_kzt_sum_3m": float(non_kzt_sum),
        "need_credit": bool(need_credit),
        "top3_spend": float(top3_spend),
        "online_spend": float(online_spend),
        "fx_count": int(fx_count),
        "travel_tx_count": int(travel_tx_count),
        "travel_ratio": float(travel_ratio)
    }

    return benefits, context

# --------------------------- Формирование push-уведомлений ---------------------------

def generate_push_text(name: str, product: str, context: Dict[str, Any]) -> str:
    short_name = str(name).split()[0] if name and isinstance(name, str) else "Клиент"

    if product == "Карта для путешествий":
        travel_3m = context.get("travel_spend_3m", 0.0)
        travel_per_month = travel_3m / 3.0
        cashback_est = travel_per_month * 0.05
        txt = f"{short_name}, в среднем вы тратите {format_amount(travel_per_month)} в месяц на поездки и отели. С тревел-картой ≈{format_amount(cashback_est)} возврата в мес. Открыть карту в приложении."
    elif product == "Премиальная карта":
        avg_bal = context.get("avg_balance", 0.0)
        txt = f"{short_name}, у вас средний остаток {format_amount(avg_bal)}. Премиальная карта даст повышенный кешбэк, страховки и привилегии. Подключить сейчас."
    elif product == "Кредитная карта":
        top = context.get("top_cats", [])[:3]
        cats = ", ".join(top) if len(top) > 0 else "ваши любимые категории"
        txt = f"{short_name}, ваши топ-категории — {cats}. Кредитная карта даёт рассрочку и повышенный кешбэк в любимых категориях. Оформить карту."
    elif product == "Обмен валют":
        cur = "USD/EUR" if context.get("non_kzt_sum_3m", 0) > 0 else "валюты"
        txt = f"{short_name}, вы часто платите в {cur}. В приложении удобный обмен без комиссии и мультивалютные счета. Настроить обмен."
    elif product == "Кредит наличными":
        txt = f"{short_name}, если нужен запас на крупные траты — можно оформить кредит наличными онлайн без справок. Узнать доступный лимит."
    elif product == "Депозит Мультивалютный":
        txt = f"{short_name}, разместите свободные средства на мультивалютном депозите — ставка выгоднее и доступ к валютным операциям. Открыть вклад."
    elif product == "Депозит Сберегательный":
        txt = f"{short_name}, для стабильного дохода — сберегательный вклад со ставкой до 16,5%. Посмотреть условия."
    elif product == "Депозит Накопительный":
        txt = f"{short_name}, если планомерно откладываете — накопительный вклад поможет копить с удобными пополнениями. Открыть вклад."
    elif product == "Инвестиции":
        txt = f"{short_name}, попробуйте инвестиции с низким порогом и без комиссий для старта. Открыть инвестиционный счёт."
    elif product == "Золотые слитки":
        txt = f"{short_name}, диверсифицируйте портфель — золотые слитки доступны в приложении и отделениях. Посмотреть предложения."
    else:
        txt = f"{short_name}, у нас есть подходящие продукты — откройте приложение для подробностей."

    
    if len(txt) > 220:
        txt = txt[:217].rstrip() + "."
    if txt.count("!") > 1:
        txt = txt.replace("!", ".", txt.count("!") - 1)
    return txt


def build_features(clients_df: pd.DataFrame, transactions: pd.DataFrame, transfers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    top_cats_for_features = ['Путешествия','Такси','Отели','Кафе и рестораны','Продукты питания','АЗС','Ювелирные украшения','Мебель','Онлайн-покупки','Обмен валют','Международные переводы']

    for _, c in clients_df.iterrows():
        cc = int(c['client_code'])
        cli_tx = transactions[transactions['client_code'] == cc] if not transactions.empty else pd.DataFrame(columns=['amount','category','currency','date','client_code'])
        cli_xf = transfers[transfers['client_code'] == cc] if not transfers.empty else pd.DataFrame(columns=['amount','type','currency','date','client_code'])

        total_spend_3m = float(cli_tx['amount'].sum()) if len(cli_tx) > 0 else 0.0
        avg_monthly_spend = total_spend_3m / 3.0 if total_spend_3m > 0 else 0.0

        cat_sums = {}
        if not cli_tx.empty:
            cat_sums = cli_tx.groupby('category')['amount'].sum().to_dict()

        cat_features = {}
        for cat in top_cats_for_features:
            val = float(cat_sums.get(cat, 0.0))
            share = float(val / (total_spend_3m + 1e-9))

            cat_features[f"cat_sum_{cat}"] = val
            cat_features[f"cat_share_{cat}"] = share

        n_tx = len(cli_tx)
        n_xf = len(cli_xf)
        n_fx = len(cli_xf[cli_xf.get('type','').isin(['fx_buy','fx_sell','deposit_fx_topup_out','deposit_fx_withdraw_in','swift_in','swift_out'])]) if not cli_xf.empty else 0
        n_invest = len(cli_xf[cli_xf.get('type','').isin(['invest_in','invest_out'])]) if not cli_xf.empty else 0
        n_gold = len(cli_xf[cli_xf.get('type','').isin(['gold_buy_out','gold_sell_in'])]) if not cli_xf.empty else 0
        large_spend_count = int((cli_tx['amount'] > 500_000).sum()) if 'amount' in cli_tx.columns else 0
        avg_balance = float(c.get('avg_monthly_balance_KZT', 0.0))

        mean_tx = float(cli_tx['amount'].mean()) if n_tx > 0 else 0.0
        med_tx = float(cli_tx['amount'].median()) if n_tx > 0 else 0.0
        non_kzt_sum = float(cli_tx[cli_tx.get('currency','KZT') != 'KZT']['amount'].sum()) if 'currency' in cli_tx.columns else 0.0

       
        travel_count = int(cli_tx[cli_tx.get('category','').isin(['Путешествия','Отели','Такси'])].shape[0]) if not cli_tx.empty else 0
        azs_share = float(cat_sums.get('АЗС', 0.0) / (total_spend_3m + 1e-9))

        row = {
            'client_code': cc,
            'total_spend_3m': total_spend_3m,
            'avg_monthly_spend': avg_monthly_spend,
            'n_tx': n_tx,
            'n_xf': n_xf,
            'n_fx': n_fx,
            'n_invest': n_invest,
            'n_gold': n_gold,
            'large_spend_count': large_spend_count,
            'avg_balance': avg_balance,
            'mean_tx': mean_tx,
            'med_tx': med_tx,
            'non_kzt_sum_3m': non_kzt_sum,
            'travel_count': travel_count,
            'azs_share': azs_share
        }
        row.update(cat_features)
        rows.append(row)

    feats = pd.DataFrame(rows).set_index('client_code').sort_index()
    return feats


def build_targets_by_heuristic(clients_df: pd.DataFrame, transactions: pd.DataFrame, transfers: pd.DataFrame, products_list: List[str]) -> pd.DataFrame:
    ys = []
    for _, c in clients_df.iterrows():
        cc = int(c['client_code'])
        benefits, _ = evaluate_benefits_for_client(c, transactions, transfers)
        row = {'client_code': cc}
        for p in products_list:
            row[p] = float(benefits.get(p, 0.0))
        ys.append(row)
    ydf = pd.DataFrame(ys).set_index('client_code').sort_index()
    return ydf


def train_and_apply_nn(clients_file: str, trans_file: str, xfer_file: str, out_dir: str):
    ensure_dir(out_dir)
    clients_df = pd.read_csv(clients_file)
    transactions = safe_read_csv(trans_file)
    transfers = safe_read_csv(xfer_file)

    if 'date' in transactions.columns:
        transactions['date'] = pd.to_datetime(transactions['date'], errors='coerce').dt.date
    if 'date' in transfers.columns:
        transfers['date'] = pd.to_datetime(transfers['date'], errors='coerce').dt.date

    # 1) X, Y
    X = build_features(clients_df, transactions, transfers)
    Y = build_targets_by_heuristic(clients_df, transactions, transfers, PRODUCTS)

    common_idx = X.index.intersection(Y.index)
    X = X.loc[common_idx]
    Y = Y.loc[common_idx]

    if len(X) >= 3:
        client_ids = list(X.index)
        train_clients, test_clients = train_test_split(client_ids, test_size=0.2, random_state=42)
        X_train = X.loc[train_clients]
        Y_train = Y.loc[train_clients]
        X_test = X.loc[test_clients]
        Y_test = Y.loc[test_clients]
    else:
        
        X_train, Y_train, X_test, Y_test = X, Y, X, Y

    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='lbfgs', max_iter=250000, random_state=42)
    mor = MultiOutputRegressor(mlp, n_jobs=1)
    pipeline = Pipeline([('scaler', StandardScaler()), ('model', mor)])

    if len(X_train) >= 2:
        n_splits = min(5, len(X_train))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        r2_per_fold = []
        print(f"Запускаю {n_splits}-fold cross-validation для оценки R2 на псевдо-таргетах (на train-сете)...")
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_train), 1):
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[test_idx]
            Y_train_fold, Y_val_fold = Y_train.iloc[train_idx], Y_train.iloc[test_idx]
            pipeline.fit(X_train_fold, Y_train_fold)
            pred = pipeline.predict(X_val_fold)
            r2s = []
            for i in range(Y_train.shape[1]):
                try:
                    r2s.append(r2_score(Y_val_fold.iloc[:, i], pred[:, i]))
                except Exception:
                    r2s.append(float('nan'))
            r2_per_fold.append(r2s)
            mean_r2 = np.nanmean(r2s)
            print(f" Fold {fold_idx}: mean R2 across products = {mean_r2:.3f}")

        mean_r2_by_product = np.nanmean(r2_per_fold, axis=0)
        print("Mean R2 per product (diagnostic on train):")
        for prod, r in zip(Y.columns, mean_r2_by_product):
            print(f"  {prod}: {r:.3f}")
    else:
        print("Недостаточно клиентов в train-сете для кросс-валидации; пропускаю CV.")

    pipeline.fit(X_train, Y_train)
    Y_pred = pipeline.predict(X)
    Y_pred_df = pd.DataFrame(Y_pred, index=X.index, columns=Y.columns)

    rec_rows = []
    debug_rows = []
    heur_comp_rows = []

    for cc in X.index:
        preds = Y_pred_df.loc[cc].sort_values(ascending=False)
        top4_model = list(preds.index[:4])
        best_model = top4_model[0]

        heur = Y.loc[cc].sort_values(ascending=False)
        top4_heur = list(heur.index[:4])

        client_row = clients_df.loc[clients_df['client_code'] == cc].iloc[0]
        _, context = evaluate_benefits_for_client(client_row, transactions, transfers)

        if best_model == "Кредитная карта" and not context.get("need_credit", False):
            replacement = None
            if context.get("travel_spend_3m", 0.0) > 10_000 and "Карта для путешествий" in preds.index:
                replacement = "Карта для путешествий"
            if replacement is None and context.get("avg_balance", 0.0) >= 1_000_000 and "Премиальная карта" in preds.index:
                replacement = "Премиальная карта"
            if replacement is None:
                for p in preds.index:
                    if p != "Кредит наличными" and p != "Кредитная карта":
                        replacement = p
                        break
            if replacement is None:
                replacement = best_model
            best_model = replacement

        push_text = generate_push_text(client_row.get('name', f'Клиент {cc}'), best_model, context)

        rec_rows.append({'client_code': int(cc), 'product': best_model, 'push_notification': push_text})
        debug_rows.append({'client_code': int(cc), 'model_top4': top4_model, 'heur_top4': top4_heur, 'predicted_benefits': json_friendly(Y_pred_df.loc[cc].to_dict()), 'context': json_friendly(context)})
        heur_comp_rows.append({'client_code': int(cc), 'heur_top4': ";".join(top4_heur), 'model_top4': ";".join(top4_model)})

    rec_df = pd.DataFrame(rec_rows)
    rec_out = os.path.join(out_dir, "recommendations_model.csv")
    rec_df.to_csv(rec_out, index=False, encoding='utf-8-sig')

    pd.DataFrame(heur_comp_rows).to_csv(os.path.join(out_dir, "metrics_model_vs_heuristic.csv"), index=False, encoding='utf-8-sig')
    with open(os.path.join(out_dir, "debug_recommendations.json"), "w", encoding='utf-8') as f:
        json.dump(json_friendly(debug_rows), f, ensure_ascii=False, indent=2)

    hits1 = sum(1 for r in debug_rows if r['model_top4'][0] in r['heur_top4'][:1])
    hits4 = sum(1 for r in debug_rows if any(p in r['heur_top4'] for p in r['model_top4'][:4]))
    n = len(debug_rows)
    hit1_rate = hits1 / n if n > 0 else 0.0
    hit4_rate = hits4 / n if n > 0 else 0.0

    print(f"Model vs Heuristic hit@1: {hits1}/{n} = {hit1_rate:.3f}")
    print(f"Model vs Heuristic hit@4: {hits4}/{n} = {hit4_rate:.3f}")

    print("Сохранены файлы:")
    print("-", rec_out)
    print("-", os.path.join(out_dir, "metrics_model_vs_heuristic.csv"))
    print("-", os.path.join(out_dir, "debug_recommendations.json"))

    return pipeline, X, Y, Y_pred_df

# --------------------------- Слияние ---------------------------

def merge_per_client_files(raw_dir: str, clients_csv: str, out_dir: str) -> Tuple[str,str]:
    ensure_dir(out_dir)
    clients_df = safe_read_csv(clients_csv)
    if clients_df.empty:
        raise SystemExit(f"clients.csv пуст или не найден: {clients_csv}")
    if 'client_code' not in clients_df.columns:
        raise SystemExit("В clients.csv обязателен столбец 'client_code'.")

    trans_patterns = [os.path.join(raw_dir, "*_transactions*.csv"), os.path.join(raw_dir, "*transactions*.csv")]
    xfer_patterns = [os.path.join(raw_dir, "*_transfers*.csv"), os.path.join(raw_dir, "*transfers*.csv"), os.path.join(raw_dir, "*transfer*.csv")]

    trans_files = sorted(set(sum((glob(p) for p in trans_patterns), [])))
    xfer_files = sorted(set(sum((glob(p) for p in xfer_patterns), [])))

    merged_trans = []
    merged_xfer = []

    for f in trans_files:
        df = safe_read_csv(f)
        if df.empty:
            continue
        df.columns = [c.strip() for c in df.columns]
        if 'client_code' not in df.columns and 'client_id' in df.columns:
            df = df.rename(columns={'client_id': 'client_code'})
        if 'client_code' not in df.columns:
            cc = find_client_code_in_filename(f)
            if cc is None:
                print(f"Не удалось определить client_code для {f}, пропускаю.")
                continue
            df['client_code'] = cc
        try:
            df['client_code'] = df['client_code'].astype(int)
        except Exception:
            def extract_int_safe(x):
                m = re.search(r"(\d+)", str(x))
                return int(m.group(1)) if m else None
            df['client_code'] = df['client_code'].apply(extract_int_safe)
        merged_trans.append(df)

    for f in xfer_files:
        df = safe_read_csv(f)
        if df.empty:
            continue
        df.columns = [c.strip() for c in df.columns]
        if 'client_code' not in df.columns and 'client_id' in df.columns:
            df = df.rename(columns={'client_id': 'client_code'})
        if 'client_code' not in df.columns:
            cc = find_client_code_in_filename(f)
            if cc is None:
                print(f"Не удалось определить client_code для {f}, пропускаю.")
                continue
            df['client_code'] = cc
        try:
            df['client_code'] = df['client_code'].astype(int)
        except Exception:
            def extract_int_safe(x):
                m = re.search(r"(\d+)", str(x))
                return int(m.group(1)) if m else None
            df['client_code'] = df['client_code'].apply(extract_int_safe)
        merged_xfer.append(df)

    trans_all_path = os.path.join(out_dir, "transactions_all.csv")
    xfer_all_path = os.path.join(out_dir, "transfers_all.csv")

    if merged_trans:
        all_trans_df = pd.concat(merged_trans, ignore_index=True, sort=False)
        if 'date' in all_trans_df.columns:
            all_trans_df['date'] = pd.to_datetime(all_trans_df['date'], errors='coerce').dt.date
        all_trans_df.to_csv(trans_all_path, index=False, encoding='utf-8-sig')
        print(f"Saved merged transactions -> {trans_all_path} (rows: {len(all_trans_df)})")
    else:
        print("Нет файлов транзакций для объединения.")

    if merged_xfer:
        all_xfer_df = pd.concat(merged_xfer, ignore_index=True, sort=False)
        if 'date' in all_xfer_df.columns:
            all_xfer_df['date'] = pd.to_datetime(all_xfer_df['date'], errors='coerce').dt.date
        all_xfer_df.to_csv(xfer_all_path, index=False, encoding='utf-8-sig')
        print(f"Saved merged transfers -> {xfer_all_path} (rows: {len(all_xfer_df)})")
    else:
        print("Нет файлов трансферов для объединения.")

    
    client_codes_profile = set(pd.read_csv(clients_csv)['client_code'].astype(int).unique())
    client_codes_data = set()
    if merged_trans:
        client_codes_data |= set(all_trans_df['client_code'].dropna().astype(int).unique())
    if merged_xfer:
        client_codes_data |= set(all_xfer_df['client_code'].dropna().astype(int).unique())
    missing = client_codes_data - client_codes_profile
    if missing:
        print(f"Внимание: коды в данных отсутствуют в clients.csv: {sorted(missing)}")
    else:
        print("Сверка client_code пройдена.")
    return (trans_all_path if merged_trans else "", xfer_all_path if merged_xfer else "")

# --------------------------- CLI / main ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Weak-supervision NN pipeline for product recommendations (expanded, full)")
    p.add_argument("--raw", default="raw", help="Папка с per-client файлами (raw/)")
    p.add_argument("--clients", default="raw/clients.csv", help="Путь к clients.csv")
    p.add_argument("--out", default="output", help="Папка вывода")
    p.add_argument("--merge-only", action="store_true", help="Только объединить файлы и выйти")
    p.add_argument("--regenerate-synthetic", action="store_true", help="Сгенерировать синтетические данные (для теста)")
    return p.parse_args()

def main():
    args = parse_args()
    ensure_dir(args.out)

    if args.regenerate_synthetic:
        raw_dir_for_gen = os.path.dirname(args.clients) or args.raw or "raw"
        if not os.path.exists(args.clients):
            print("Генерирую синтетические данные...")
            generate_synthetic(raw_dir_for_gen, args.clients, n_clients=NUM_CLIENTS)
        else:
            print("clients.csv уже существует — если хотите пересоздать, удалите файл и запустите снова с --regenerate-synthetic")

    print("1) Объединяю per-client файлы (если есть)...")
    trans_all, xfer_all = merge_per_client_files(args.raw, args.clients, args.out)

    if args.merge_only:
        print("Опция --merge-only: завершил объединение.")
        return

    if not trans_all:
        cand = os.path.join(args.raw, "transactions_all.csv")
        trans_all = cand if os.path.exists(cand) else ""
    if not xfer_all:
        cand = os.path.join(args.raw, "transfers_all.csv")
        xfer_all = cand if os.path.exists(cand) else ""

    if not trans_all:
        empty_trans = pd.DataFrame(columns=['date','category','amount','currency','client_code'])
        trans_all = os.path.join(args.out, 'transactions_all.csv')
        empty_trans.to_csv(trans_all, index=False, encoding='utf-8-sig')
        print(f"Создан пустой файл транзакций -> {trans_all}")
    if not xfer_all:
        empty_xfer = pd.DataFrame(columns=['date','type','direction','amount','currency','client_code'])
        xfer_all = os.path.join(args.out, 'transfers_all.csv')
        empty_xfer.to_csv(xfer_all, index=False, encoding='utf-8-sig')
        print(f"Создан пустой файл трансферов -> {xfer_all}")

    print("2) Обучаю нейросеть на псевдо-таргетах и формирую рекомендации...")
    pipeline, X, Y, Y_pred = train_and_apply_nn(args.clients, trans_all, xfer_all, args.out)
    print("Готово. Результаты в папке:", args.out)

if __name__ == "__main__":
    main()

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Стартовый датасет мерчантов
train_records = [
    # Їжа
    ("Сільпо Київ", "їжа"),
    ("АТБ Маркет", "їжа"),
    ("Novus Супермаркет", "їжа"),
    ("McDonalds Pochayna", "їжа"),
    ("KFC Express", "їжа"),
    ("Пузата Хата", "їжа"),
    ("Кав'ярня Арома Кава", "їжа"),
    ("Пекарня Львівські круасани", "їжа"),
    ("Glovo delivery food", "їжа"),
    ("Bolt Food кур'єр", "їжа"),
    ("Fora магазин", "їжа"),
    ("Auchan гіпермаркет", "їжа"),
    # Транспорт
    ("Uber trip Kyiv", "транспорт"),
    ("Uklon поїздка таксі", "транспорт"),
    ("Bolt Taxi", "транспорт"),
    ("WOG АЗС паливо", "транспорт"),
    ("OKKO бензин А-95", "транспорт"),
    ("Київський метрополітен", "транспорт"),
    ("Укрзалізниця квитки", "транспорт"),
    ("Паркінг City Center", "транспорт"),
    ("SOCAR дизель", "транспорт"),
    # Комунальні
    ("YASNO електроенергія", "комунальні"),
    ("Київводоканал оплата", "комунальні"),
    ("Київтеплоенерго опалення", "комунальні"),
    ("Нафтогаз газ для дому", "комунальні"),
    ("Комунальні послуги ГЕРЦ", "комунальні"),
    ("Інтернет Lanet абонплата", "комунальні"),
    ("Київстар домашній інтернет", "комунальні"),
    ("ОСББ внесок за квартиру", "комунальні"),
    # Покупки
    ("Zara одяг ТРЦ", "покупки"),
    ("Rozetka маркетплейс", "покупки"),
    ("Reserved одяг Ocean Plaza", "покупки"),
    ("Епіцентр К інструменти", "покупки"),
    ("H&M одяг взуття", "покупки"),
    ("Comfy техніка", "покупки"),
    ("Аптека Низьких Цін ліки", "покупки"),
    ("EVA косметика побут", "покупки"),
    ("Makeup інтернет магазин", "покупки"),
    ("JYSK декор для дому", "покупки"),
    # Розваги
    ("Multiplex Lavina кінотеатр", "розваги"),
    ("Планета Кіно IMAX", "розваги"),
    ("Steam games purchase", "розваги"),
    ("PlayStation Network games", "розваги"),
    ("Netflix щомісячна підписка", "розваги"),
    ("Spotify Premium музика", "розваги"),
    ("Atlas клуб квитки концерт", "розваги"),
    ("Боулінг клуб Блокбастер", "розваги"),
    ("Книгарня Є книга", "розваги"),
]

df = pd.DataFrame(train_records, columns=["description", "category"])

pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
        ("clf", LinearSVC(C=1.0, random_state=42)),
    ]
)

print("Тренування моделі класифікації...")
pipeline.fit(df["description"], df["category"])

joblib.dump(pipeline, "classifier.joblib")
print("✅ Модель успішно збережено у 'classifier.joblib'")
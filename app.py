import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

PNEUMONIA_MODEL = BASE_DIR / "pneumonia_best_model.keras"
TUBERCULOSIS_MODEL = MODELS_DIR / "tuberculosis_optimized.keras"
STUDENT_MODEL = MODELS_DIR / "student_performance_optimized.keras"
STUDENT_PREPROCESSING = MODELS_DIR / "student_preprocessing.json"
BANK_MODEL = BASE_DIR / "DNN_Bank_optimized.keras"
BANK_PREPROCESSING = MODELS_DIR / "bank_preprocessing.json"

MEDICAL_WARNING = (
    "Cet outil est expérimental et pédagogique. "
    "Il ne remplace pas l'avis ni le diagnostic d'un professionnel de santé."
)

st.set_page_config(
    page_title="Projets Deep Learning",
    page_icon="🧠",
    layout="centered"
)


@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path, compile=False)


@st.cache_data
def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Pages images : pneumonie et tuberculose
# ---------------------------------------------------------------------------

def prepare_image(image, size):
    # Même traitement que ImageDataGenerator à l'entraînement :
    # redimensionnement "nearest" puis valeurs ramenées entre 0 et 1.
    image_resized = image.resize(size, Image.NEAREST)
    image_array = np.array(image_resized, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def image_classifier_page(title, description, model_path, image_size,
                          negative_class, positive_class):
    st.title(title)
    st.write(description)
    st.warning(MEDICAL_WARNING)

    uploaded_file = st.file_uploader(
        "Importer une radiographie thoracique",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Radiographie importée", width="stretch")

        image_array = prepare_image(image, image_size)

        if st.button("Analyser l'image"):
            with st.spinner("Analyse en cours..."):
                model = load_model(str(model_path))
                prediction = float(model.predict(image_array, verbose=0)[0][0])

            st.subheader("Résultat de la prédiction")

            if prediction >= 0.5:
                st.write(f"Classe prédite : {positive_class}")
            else:
                st.write(f"Classe prédite : {negative_class}")

            st.write(f"Score du modèle ({positive_class}) : {prediction:.2%}")


def pneumonia_page():
    image_classifier_page(
        title="🫁 Pneumonia Detection",
        description="Analyse pédagogique de radiographies thoraciques.",
        model_path=PNEUMONIA_MODEL,
        image_size=(64, 64),
        negative_class="NORMAL",
        positive_class="PNEUMONIA",
    )


def tuberculosis_page():
    image_classifier_page(
        title="🦠 Tuberculosis Detection",
        description=(
            "TP5 : détection de la tuberculose sur radiographie thoracique "
            "(transfert d'apprentissage VGG16, hyperparamètres optimisés)."
        ),
        model_path=TUBERCULOSIS_MODEL,
        image_size=(150, 150),
        negative_class="NORMAL",
        positive_class="TUBERCULOSIS",
    )


# ---------------------------------------------------------------------------
# Page TP2 : performance des étudiants (régression de la note finale G3)
# ---------------------------------------------------------------------------

YES_NO = {"yes": "Oui", "no": "Non"}
EDUCATION_LEVELS = {
    0: "Aucun", 1: "Primaire", 2: "Collège", 3: "Lycée", 4: "Supérieur"
}
JOBS = {
    "teacher": "Enseignant(e)", "health": "Santé", "services": "Services",
    "at_home": "Au foyer", "other": "Autre"
}

STUDENT_FIELDS = {
    "school": ("École", {"GP": "Gabriel Pereira", "MS": "Mousinho da Silveira"}),
    "sex": ("Sexe", {"F": "Fille", "M": "Garçon"}),
    "age": ("Âge", None),
    "address": ("Lieu d'habitation", {"U": "Urbain", "R": "Rural"}),
    "famsize": ("Taille de la famille", {"LE3": "3 personnes ou moins", "GT3": "Plus de 3 personnes"}),
    "Pstatus": ("Situation des parents", {"T": "Vivent ensemble", "A": "Séparés"}),
    "Medu": ("Niveau d'études de la mère", EDUCATION_LEVELS),
    "Fedu": ("Niveau d'études du père", EDUCATION_LEVELS),
    "Mjob": ("Profession de la mère", JOBS),
    "Fjob": ("Profession du père", JOBS),
    "reason": ("Raison du choix de l'école", {
        "home": "Proche du domicile", "reputation": "Réputation",
        "course": "Programme", "other": "Autre"
    }),
    "guardian": ("Tuteur", {"mother": "Mère", "father": "Père", "other": "Autre"}),
    "traveltime": ("Temps de trajet", {1: "Moins de 15 min", 2: "15 à 30 min", 3: "30 min à 1 h", 4: "Plus d'1 h"}),
    "studytime": ("Temps d'étude par semaine", {1: "Moins de 2 h", 2: "2 à 5 h", 3: "5 à 10 h", 4: "Plus de 10 h"}),
    "failures": ("Nombre d'échecs passés", None),
    "schoolsup": ("Soutien scolaire de l'école", YES_NO),
    "famsup": ("Soutien scolaire de la famille", YES_NO),
    "paid": ("Cours particuliers payants", YES_NO),
    "activities": ("Activités extrascolaires", YES_NO),
    "nursery": ("A fréquenté la maternelle", YES_NO),
    "higher": ("Veut faire des études supérieures", YES_NO),
    "internet": ("Internet à la maison", YES_NO),
    "romantic": ("En couple", YES_NO),
    "famrel": ("Relations familiales (1 = très mauvaises, 5 = excellentes)", None),
    "freetime": ("Temps libre après l'école (1 à 5)", None),
    "goout": ("Sorties avec les amis (1 à 5)", None),
    "Dalc": ("Alcool en semaine (1 à 5)", None),
    "Walc": ("Alcool le week-end (1 à 5)", None),
    "health": ("État de santé (1 = très mauvais, 5 = très bon)", None),
    "absences": ("Nombre d'absences", None),
    "G1": ("Note du 1er trimestre (G1, sur 20)", None),
    "G2": ("Note du 2e trimestre (G2, sur 20)", None),
}

STUDENT_SECTIONS = {
    "Notes de l'année": ["G1", "G2"],
    "Profil": ["school", "sex", "age", "address", "famsize", "Pstatus", "guardian", "reason"],
    "Famille": ["Medu", "Fedu", "Mjob", "Fjob", "famrel", "famsup"],
    "Scolarité": ["studytime", "traveltime", "failures", "absences", "schoolsup",
                  "paid", "activities", "nursery", "higher", "internet"],
    "Vie sociale et santé": ["romantic", "freetime", "goout", "Dalc", "Walc", "health"],
}


def student_input(feature):
    label, options = STUDENT_FIELDS[feature["name"]]

    if feature["type"] == "categorical":
        choices = feature["choices"]
        return st.selectbox(
            label, choices,
            index=choices.index(feature["default"]),
            format_func=lambda value: options.get(value, value) if options else value,
        )

    low, high, default = feature["min"], feature["max"], feature["default"]
    if options:
        choices = list(range(low, high + 1))
        return st.selectbox(
            label, choices,
            index=choices.index(default),
            format_func=lambda value: options.get(value, str(value)),
        )
    if high - low <= 25:
        return st.slider(label, low, high, default)
    return st.number_input(label, min_value=low, max_value=high, value=default, step=1)


def encode_student(values, preprocessing):
    row = []
    for column in preprocessing["encoded"]:
        if "value" in column:
            row.append(1.0 if str(values[column["source"]]) == column["value"] else 0.0)
        else:
            row.append(float(values[column["source"]]))
    row = (np.array(row) - np.array(preprocessing["mean"])) / np.array(preprocessing["scale"])
    return row.reshape(1, -1).astype(np.float32)


def student_page():
    st.title("🎓 Student Performance")
    st.write(
        "TP2 : prédiction de la note finale en mathématiques (G3, sur 20) "
        "d'un élève à partir de son profil et de ses notes de l'année "
        "(jeu de données UCI Student Performance)."
    )

    preprocessing = load_json(str(STUDENT_PREPROCESSING))
    features = {f["name"]: f for f in preprocessing["raw_features"]}

    values = {}
    with st.form("student_form"):
        for index, (section, names) in enumerate(STUDENT_SECTIONS.items()):
            with st.expander(section, expanded=(index == 0)):
                columns = st.columns(2)
                for position, name in enumerate(names):
                    with columns[position % 2]:
                        values[name] = student_input(features[name])
        submitted = st.form_submit_button("Prédire la note finale")

    if submitted:
        model = load_model(str(STUDENT_MODEL))
        prediction = float(model.predict(encode_student(values, preprocessing), verbose=0)[0][0])
        prediction = min(max(prediction, 0.0), 20.0)

        st.subheader("Résultat de la prédiction")
        st.metric("Note finale prédite (G3)", f"{prediction:.1f} / 20")
        st.caption("Sur le jeu de test, le modèle se trompe en moyenne d'environ 2 points.")


# ---------------------------------------------------------------------------
# Page DNN Bank : souscription à un dépôt à terme
# ---------------------------------------------------------------------------

BANK_CATEGORY_LABELS = {
    "job": ("Profession", {
        "admin.": "Administratif", "blue-collar": "Ouvrier", "entrepreneur": "Entrepreneur",
        "housemaid": "Employé(e) de maison", "management": "Cadre", "retired": "Retraité(e)",
        "self-employed": "Indépendant(e)", "services": "Services", "student": "Étudiant(e)",
        "technician": "Technicien(ne)", "unemployed": "Sans emploi", "unknown": "Inconnu",
    }),
    "marital": ("Situation familiale", {
        "divorced": "Divorcé(e)", "married": "Marié(e)", "single": "Célibataire", "unknown": "Inconnu",
    }),
    "education": ("Niveau d'études", {
        "basic.4y": "Primaire (4 ans)", "basic.6y": "Primaire (6 ans)", "basic.9y": "Collège (9 ans)",
        "high.school": "Lycée", "illiterate": "Non scolarisé(e)",
        "professional.course": "Formation professionnelle", "university.degree": "Diplôme universitaire",
        "unknown": "Inconnu",
    }),
    "default": ("Défaut de paiement", {"no": "Non", "yes": "Oui", "unknown": "Inconnu"}),
    "housing": ("Prêt immobilier", {"no": "Non", "yes": "Oui", "unknown": "Inconnu"}),
    "loan": ("Prêt personnel", {"no": "Non", "yes": "Oui", "unknown": "Inconnu"}),
    "contact": ("Type de contact", {"cellular": "Mobile", "telephone": "Fixe"}),
    "month": ("Mois du dernier contact", {
        "jan": "Janvier", "feb": "Février", "mar": "Mars", "apr": "Avril", "may": "Mai", "jun": "Juin",
        "jul": "Juillet", "aug": "Août", "sep": "Septembre", "oct": "Octobre", "nov": "Novembre",
        "dec": "Décembre",
    }),
    "day_of_week": ("Jour du dernier contact", {
        "mon": "Lundi", "tue": "Mardi", "wed": "Mercredi", "thu": "Jeudi", "fri": "Vendredi",
    }),
    "poutcome": ("Résultat de la campagne précédente", {
        "failure": "Échec", "nonexistent": "Pas de campagne précédente", "success": "Succès",
    }),
}

BANK_NUMERIC_LABELS = {
    "age": "Âge",
    "duration": "Durée du dernier appel (secondes)",
    "campaign": "Nombre de contacts pendant cette campagne",
    "pdays": "Jours depuis le dernier contact (999 = jamais contacté)",
    "previous": "Nombre de contacts avant cette campagne",
}


def bank_page():
    st.title("🏦 Bank Marketing")
    st.write(
        "Prédiction de la souscription d'un client à un dépôt à terme "
        "après une campagne de marketing téléphonique (réseau de neurones dense optimisé)."
    )

    if not BANK_PREPROCESSING.exists():
        st.info(
            "Cette page n'est pas encore disponible : il manque les paramètres de "
            "normalisation du modèle (fichier `models/bank_preprocessing.json`). "
            "Ils se génèrent avec `tools/build_preprocessing.py` à partir du fichier "
            "`Copie de banking_dataset.csv` utilisé dans le notebook."
        )
        return

    preprocessing = load_json(str(BANK_PREPROCESSING))
    categories = preprocessing["categories"]

    values = {}
    with st.form("bank_form"):
        columns = st.columns(2)
        for position, name in enumerate(preprocessing["features"]):
            with columns[position % 2]:
                if name in categories:
                    label, translations = BANK_CATEGORY_LABELS[name]
                    # Ordre d'affichage naturel (mois, jours) ; l'encodage reste celui du notebook
                    options = [v for v in translations if v in categories[name]]
                    options += [v for v in categories[name] if v not in options]
                    default = preprocessing.get("category_defaults", {}).get(name)
                    values[name] = st.selectbox(
                        label, options,
                        index=options.index(default) if default in options else 0,
                        format_func=lambda value, t=translations: t.get(value, value),
                    )
                else:
                    values[name] = st.number_input(
                        BANK_NUMERIC_LABELS[name], min_value=0,
                        value=preprocessing["numeric_defaults"][name], step=1,
                    )
        submitted = st.form_submit_button("Prédire la souscription")

    if submitted:
        row = [
            categories[name].index(values[name]) if name in categories else values[name]
            for name in preprocessing["features"]
        ]
        row = (np.array(row, dtype=np.float64) - np.array(preprocessing["center"])) / np.array(preprocessing["scale"])

        model = load_model(str(BANK_MODEL))
        prediction = float(model.predict(row.reshape(1, -1).astype(np.float32), verbose=0)[0][0])

        st.subheader("Résultat de la prédiction")
        if prediction >= 0.5:
            st.write("Le client va probablement **souscrire**.")
        else:
            st.write("Le client ne va probablement **pas souscrire**.")
        st.write(f"Probabilité de souscription : {prediction:.2%}")


# ---------------------------------------------------------------------------
# Accueil et navigation
# ---------------------------------------------------------------------------

def home_page():
    st.title("🧠 Projets Deep Learning")
    st.write(
        "Cette application réunit les modèles entraînés pendant les TP. "
        "Choisissez un projet dans le menu à gauche ou ci-dessous."
    )
    for page, summary in [
        (PAGES["pneumonia"], "Classification de radiographies thoraciques (CNN, images 64×64)."),
        (PAGES["tuberculosis"], "TP5 : radiographies thoraciques, transfert d'apprentissage VGG16."),
        (PAGES["student"], "TP2 : prédiction de la note finale d'un élève (DNN, régression)."),
        (PAGES["bank"], "Prédiction de la souscription d'un client bancaire (DNN, classification)."),
    ]:
        st.page_link(page)
        st.caption(summary)


PAGES = {
    "home": st.Page(home_page, title="Accueil", icon="🏠", url_path="accueil", default=True),
    "pneumonia": st.Page(pneumonia_page, title="Pneumonie", icon="🫁", url_path="pneumonie"),
    "tuberculosis": st.Page(tuberculosis_page, title="Tuberculose", icon="🦠", url_path="tuberculose"),
    "student": st.Page(student_page, title="Performance des étudiants", icon="🎓", url_path="etudiants"),
    "bank": st.Page(bank_page, title="Marketing bancaire", icon="🏦", url_path="banque"),
}

st.navigation(list(PAGES.values())).run()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from rich import print as rprint
from tabulate import tabulate
import io


class Preprocesador:
    def __init__(self, dataframe, columna_objetivo=None):
        self.df_original = dataframe.copy()
        self.df = dataframe.copy()
        self.columna_objetivo = columna_objetivo

        if columna_objetivo and columna_objetivo in self.df.columns:
            self.X = self.df.drop(columns=[columna_objetivo])
            self.y = self.df[columna_objetivo]
        else:
            self.X = self.df.iloc[:, :-1]
            self.y = self.df.iloc[:, -1]

    def _sincronizar_df(self):
        if self.y is not None:
            self.df = pd.concat([self.X, self.y], axis=1)
        else:
            self.df = self.X.copy()

    def explorar_datos(self):
        rprint("[bold cyan]\n Primeras filas del dataset:[/bold cyan]")
        print(tabulate(self.df.head(), headers="keys", tablefmt="pretty"))

        rprint("[bold cyan]\n Info del dataset:[/bold cyan]")
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        print(buffer.getvalue())

        rprint("[bold cyan]\n Estadísticas descriptivas:[/bold cyan]")
        print(tabulate(self.df.describe(), headers="keys", tablefmt="pretty"))

        rprint("[bold cyan]\n Valores nulos por columna:[/bold cyan]")
        print(
            tabulate(
                self.df.isnull().sum().to_frame(name="Valores nulos"),
                headers="keys",
                tablefmt="pretty",
            )
        )

        rprint("[bold cyan]\n Valores duplicados:[/bold cyan]")
        print(f"Duplicados: {self.df.duplicated().sum()}")

        plt.figure(figsize=(10, 4))
        sns.heatmap(self.df.isnull(), cbar=False, cmap="viridis")
        plt.title("Mapa de calor de valores nulos")
        plt.show()

        # If columna_objetivo is Categorical, plot its distribution
        if self.df[self.columna_objetivo].dtype == "object":
            rprint(
                "[bold cyan]\n Distribución de clases en la variable objetivo:[/bold cyan]"
            )
            if self.columna_objetivo and self.columna_objetivo in self.df.columns:
                self.df[self.columna_objetivo].value_counts().plot(
                    kind="bar", color="skyblue"
                )
                plt.title("Distribución de clases")
                plt.xlabel("Clase")
                plt.ylabel("Frecuencia")
                plt.show()

        cat_cols, num_cols = self.obtener_tipos_columnas()
        self.df[num_cols].hist(bins=30, figsize=(15, 10))
        plt.tight_layout()
        plt.show()

        z_scores = (self.df[num_cols] - self.df[num_cols].mean()) / self.df[
            num_cols
        ].std()
        rprint("[bold cyan]\n Outliers detectados por z-score (>3):[/bold cyan]")
        print(
            tabulate(
                (np.abs(z_scores) > 3).sum().to_frame(name="Outliers (z>3)"),
                headers="keys",
                tablefmt="pretty",
            )
        )
        self.df[num_cols].hist(bins=30, figsize=(15, 10))
        plt.tight_layout()
        plt.show()

    def obtener_tipos_columnas(self):
        cat_cols = self.X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = self.X.select_dtypes(include=["number"]).columns.tolist()

        rprint("[bold yellow]\n Columnas categóricas:[/bold yellow]")
        print(cat_cols)
        rprint("[bold green]\n Columnas numéricas:[/bold green]")
        print(num_cols)

        return cat_cols, num_cols

    def limpiar_datos(self):
        self.df.drop_duplicates(inplace=True)
        self.df.fillna(self.df.median(numeric_only=True), inplace=True)
        if self.columna_objetivo and self.columna_objetivo in self.df.columns:
            self.X = self.df.drop(columns=[self.columna_objetivo])
            self.y = self.df[self.columna_objetivo]
        else:
            self.X = self.df.iloc[:, :-1]
            self.y = self.df.iloc[:, -1]

    def manejar_outliers(self, columnas):
        for col in columnas:
            Q1 = self.X[col].quantile(0.25)
            Q3 = self.X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            mediana = self.X[col].median()
            outliers = self.X[(self.X[col] < lower) | (self.X[col] > upper)][col]
            if not outliers.empty:
                rprint(
                    f"[bold magenta]\n Outliers en '{col}': {len(outliers)} reemplazados por {mediana}"
                )
                rprint(f" Valores fuera de rango: {outliers.values}")
                print(f"  Rango: [{lower}, {upper}]")
                print(f"  Reemplazando outliers por mediana ({mediana})")
                self.X.loc[outliers.index, col] = mediana
        self._sincronizar_df()

    def plot_boxplot(self, columnas=None):
        if not columnas:
            columnas = self.X.select_dtypes(include=["number"]).columns.tolist()
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.X[columnas], orient="h")
        plt.title("Boxplot de variables normalizadas")
        plt.show()

    def normalizar(self, columnas=None):
        scaler = StandardScaler()
        if columnas:
            self.X[columnas] = scaler.fit_transform(self.X[columnas])
        else:
            self.X = pd.DataFrame(scaler.fit_transform(self.X), columns=self.X.columns)
        self._sincronizar_df()

    def codificar_categoricas(self):
        for col in self.X.select_dtypes(include="object").columns:
            if self.X[col].nunique() <= 10:
                self.X = pd.get_dummies(self.X, columns=[col])
            else:
                le = LabelEncoder()
                self.X[col] = le.fit_transform(self.X[col])
        self._sincronizar_df()

    def reducir_dimensionalidad(self, metodo="pca"):
        X = self.X
        y = self.y
        if metodo == "pca":
            pca = PCA(n_components=2)
            componentes = pca.fit_transform(X)
        elif metodo == "tsne":
            tsne = TSNE(n_components=2, random_state=42)
            componentes = tsne.fit_transform(X)
        else:
            raise ValueError("Método no soportado: elige 'pca' o 'tsne'")

        plt.figure(figsize=(8, 6))
        if y is not None:
            sns.scatterplot(
                x=componentes[:, 0], y=componentes[:, 1], hue=y, palette="Set2"
            )
        else:
            plt.scatter(componentes[:, 0], componentes[:, 1], alpha=0.7)
        plt.title(f"Visualización con {metodo.upper()}")
        plt.xlabel("Componente 1")
        plt.ylabel("Componente 2")
        plt.grid(True)
        plt.show()

    def balancear_clases(self, metodo="smote"):
        if metodo == "smote":
            sampler = SMOTE()
        elif metodo == "undersample":
            sampler = RandomUnderSampler()
        else:
            raise ValueError("Método no soportado: elige 'smote' o 'undersample'")

        X_resampled, y_resampled = sampler.fit_resample(self.X, self.y)
        self.X = pd.DataFrame(X_resampled, columns=self.X.columns)
        self.y = pd.Series(y_resampled, name=self.columna_objetivo)
        self._sincronizar_df()

        plt.figure(figsize=(6, 4))
        self.y.value_counts().plot(kind="bar", color="skyblue")
        plt.title("Distribución de clases después del balanceo")
        plt.xlabel("Clase")
        plt.ylabel("Frecuencia")
        plt.show()

    def dividir_datos(self, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def validacion_cruzada(self, k=5):
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        for i, (train_idx, test_idx) in enumerate(kf.split(self.df)):
            print(f"Fold {i + 1}: Train idx: {train_idx[:5]}, Test idx: {test_idx[:5]}")

    # Función para visualizar outliers
    def plot_outliers(self, columns):
        """
        Muestra boxplots de las variables numéricas para identificar valores atípicos.
        """
        plt.figure(figsize=(12, 6))
        for i, col in enumerate(columns):
            plt.subplot(1, len(columns), i + 1)
            sns.boxplot(y=self.X[col], color="orange")
            plt.title(f"Outliers en {col}")
        plt.tight_layout()
        plt.show()

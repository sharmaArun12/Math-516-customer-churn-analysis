
# T-Transformer for Customer Churn


import copy
import math
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from torch import nn
from torch.utils.data import DataLoader, Dataset


def setSeed(seedValue: int = 42) -> None:
    """
    Set random seeds so runs stay reproducible.
    """
    random.seed(seedValue)
    np.random.seed(seedValue)
    torch.manual_seed(seedValue)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seedValue)


@dataclass
class DataBundle:
    """
    Store processed splits and metadata used by the pipeline.
    """
    xTrainCat: np.ndarray
    xValCat: np.ndarray
    xTestCat: np.ndarray
    xTrainNum: np.ndarray
    xValNum: np.ndarray
    xTestNum: np.ndarray
    yTrain: np.ndarray
    yVal: np.ndarray
    yTest: np.ndarray
    categoricalColumns: list
    numericalColumns: list
    categoryCardinalities: list
    quantileTransformer: QuantileTransformer
    categoryMaps: dict


def loadAndCleanData(csvPath: str) -> pd.DataFrame:
    """
    Load churn data and apply the basic cleaning.
    """
    df = pd.read_csv(csvPath)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)

    return df


def getColumnGroups(df: pd.DataFrame) -> tuple[list, list]:
    """
    Separate feature columns into categorical and numerical groups.
    """
    categoricalColumns = df.select_dtypes(include=["object"]).columns.tolist()
    numericalColumns = [
        columnName for columnName in df.columns
        if columnName not in categoricalColumns + ["Churn"]
    ]
    return categoricalColumns, numericalColumns


def buildCategoryMaps(trainDf: pd.DataFrame, categoricalColumns: list) -> dict:
    """
    Build integer lookup tables for categorical features using training data only.
    """
    categoryMaps = {}

    for columnName in categoricalColumns:
        uniqueValues = sorted(trainDf[columnName].astype(str).unique().tolist())
        categoryMaps[columnName] = {
            value: index + 1 for index, value in enumerate(uniqueValues)
        }

    return categoryMaps


def encodeCategoricalFrame(
    df: pd.DataFrame,
    categoricalColumns: list,
    categoryMaps: dict
) -> np.ndarray:
    """
    Convert categorical values into integer indices. Unseen values map to 0.
    """
    encodedColumns = []

    for columnName in categoricalColumns:
        valueToIndex = categoryMaps[columnName]
        encoded = df[columnName].astype(str).map(
            lambda value: valueToIndex.get(value, 0)
        ).to_numpy()
        encodedColumns.append(encoded)

    if len(encodedColumns) == 0:
        return np.zeros((len(df), 0), dtype=np.int64)

    return np.column_stack(encodedColumns).astype(np.int64)


def fitQuantileTransformer(
    trainDf: pd.DataFrame,
    numericalColumns: list,
    randomState: int = 42
) -> QuantileTransformer:
    """
    Fit quantile normalization on training numerical features.
    """
    if len(numericalColumns) == 0:
        return None

    nSamples = len(trainDf)
    nQuantiles = min(1000, nSamples)

    transformer = QuantileTransformer(
        n_quantiles=nQuantiles,
        output_distribution="normal",
        random_state=randomState
    )
    transformer.fit(trainDf[numericalColumns])

    return transformer


def transformNumericalFrame(
    df: pd.DataFrame,
    numericalColumns: list,
    transformer: QuantileTransformer
) -> np.ndarray:
    """
    Apply the fitted quantile transformer to numerical features.
    """
    if len(numericalColumns) == 0:
        return np.zeros((len(df), 0), dtype=np.float32)

    transformed = transformer.transform(df[numericalColumns])
    return transformed.astype(np.float32)


def prepareDataBundle(
    df: pd.DataFrame,
    testSize: float = 0.2,
    valSize: float = 0.2,
    randomState: int = 42
) -> DataBundle:
    """
    Split the data into train, validation, and test sets, then preprocess it.
    """
    trainDf, testDf = train_test_split(
        df,
        test_size=testSize,
        stratify=df["Churn"],
        random_state=randomState
    )

    adjustedValSize = valSize / (1.0 - testSize)

    trainDf, valDf = train_test_split(
        trainDf,
        test_size=adjustedValSize,
        stratify=trainDf["Churn"],
        random_state=randomState
    )

    categoricalColumns, numericalColumns = getColumnGroups(df)
    categoryMaps = buildCategoryMaps(trainDf, categoricalColumns)
    quantileTransformer = fitQuantileTransformer(trainDf, numericalColumns, randomState)

    xTrainCat = encodeCategoricalFrame(trainDf, categoricalColumns, categoryMaps)
    xValCat = encodeCategoricalFrame(valDf, categoricalColumns, categoryMaps)
    xTestCat = encodeCategoricalFrame(testDf, categoricalColumns, categoryMaps)

    xTrainNum = transformNumericalFrame(trainDf, numericalColumns, quantileTransformer)
    xValNum = transformNumericalFrame(valDf, numericalColumns, quantileTransformer)
    xTestNum = transformNumericalFrame(testDf, numericalColumns, quantileTransformer)

    yTrain = trainDf["Churn"].to_numpy(dtype=np.float32)
    yVal = valDf["Churn"].to_numpy(dtype=np.float32)
    yTest = testDf["Churn"].to_numpy(dtype=np.float32)

    categoryCardinalities = [
        len(categoryMaps[columnName]) + 1 for columnName in categoricalColumns
    ]

    return DataBundle(
        xTrainCat=xTrainCat,
        xValCat=xValCat,
        xTestCat=xTestCat,
        xTrainNum=xTrainNum,
        xValNum=xValNum,
        xTestNum=xTestNum,
        yTrain=yTrain,
        yVal=yVal,
        yTest=yTest,
        categoricalColumns=categoricalColumns,
        numericalColumns=numericalColumns,
        categoryCardinalities=categoryCardinalities,
        quantileTransformer=quantileTransformer,
        categoryMaps=categoryMaps
    )


class ChurnDataset(Dataset):
    """
    Return one sample at a time as categorical data, numerical data, and label.
    """

    def __init__(self, xCat: np.ndarray, xNum: np.ndarray, y: np.ndarray):
        self.xCat = torch.tensor(xCat, dtype=torch.long)
        self.xNum = torch.tensor(xNum, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.xCat[index], self.xNum[index], self.y[index]


class NumericalFeatureTokenizer(nn.Module):
    """
    Turn each numerical feature into a learnable embedding token.
    """

    def __init__(self, numNumericalFeatures: int, embeddingDim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(numNumericalFeatures, embeddingDim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(numNumericalFeatures, embeddingDim))

    def forward(self, xNum: torch.Tensor) -> torch.Tensor:
        return xNum.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class CategoricalFeatureTokenizer(nn.Module):
    """
    Look up embeddings for categorical feature indices using one shared table.
    """

    def __init__(self, categoryCardinalities: list, embeddingDim: int):
        super().__init__()
        offsets = np.cumsum([0] + categoryCardinalities[:-1]).astype(np.int64)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

        totalCategories = sum(categoryCardinalities)
        self.embedding = nn.Embedding(totalCategories, embeddingDim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, xCat: torch.Tensor) -> torch.Tensor:
        if xCat.size(1) == 0:
            batchSize = xCat.size(0)
            return torch.zeros(
                batchSize, 0, self.embedding.embedding_dim, device=xCat.device
            )

        xWithOffsets = xCat + self.offsets.unsqueeze(0)
        return self.embedding(xWithOffsets)


class TransformerBlock(nn.Module):
    """
    One Transformer block with self-attention and feedforward layers.
    """

    def __init__(
        self,
        embeddingDim: int,
        numHeads: int,
        dropoutRate: float,
        ffnMultiplier: int = 4
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embeddingDim,
            num_heads=numHeads,
            dropout=dropoutRate,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embeddingDim)
        self.norm2 = nn.LayerNorm(embeddingDim)

        self.feedForward = nn.Sequential(
            nn.Linear(embeddingDim, ffnMultiplier * embeddingDim),
            nn.ReLU(),
            nn.Dropout(dropoutRate),
            nn.Linear(ffnMultiplier * embeddingDim, embeddingDim)
        )

        self.dropout = nn.Dropout(dropoutRate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attentionOut, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attentionOut))

        feedForwardOut = self.feedForward(x)
        x = self.norm2(x + self.dropout(feedForwardOut))

        return x


class FTTransformer(nn.Module):
    """
    FT-Transformer for binary churn prediction.
    """

    def __init__(
        self,
        categoryCardinalities: list,
        numNumericalFeatures: int,
        embeddingDim: int = 32,
        numHeads: int = 4,
        numBlocks: int = 3,
        dropoutRate: float = 0.1
    ):
        super().__init__()

        self.categoricalTokenizer = CategoricalFeatureTokenizer(
            categoryCardinalities, embeddingDim
        )
        self.numericalTokenizer = NumericalFeatureTokenizer(
            numNumericalFeatures, embeddingDim
        )
        self.clsToken = nn.Parameter(torch.randn(1, 1, embeddingDim) * 0.02)

        self.transformerBlocks = nn.ModuleList([
            TransformerBlock(
                embeddingDim=embeddingDim,
                numHeads=numHeads,
                dropoutRate=dropoutRate
            )
            for _ in range(numBlocks)
        ])

        self.outputHead = nn.Sequential(
            nn.LayerNorm(embeddingDim),
            nn.ReLU(),
            nn.Linear(embeddingDim, 1)
        )

    def forward(self, xCat: torch.Tensor, xNum: torch.Tensor) -> torch.Tensor:
        batchSize = xNum.size(0)

        catTokens = self.categoricalTokenizer(xCat)
        numTokens = self.numericalTokenizer(xNum)

        clsTokens = self.clsToken.expand(batchSize, -1, -1)
        allTokens = torch.cat([clsTokens, catTokens, numTokens], dim=1)

        for block in self.transformerBlocks:
            allTokens = block(allTokens)

        clsRepresentation = allTokens[:, 0, :]
        logits = self.outputHead(clsRepresentation).squeeze(1)

        return logits


def trainOneEpoch(model, dataLoader, optimizer, lossFunction, device) -> float:
    """
    Train the model for one full pass over the training set.
    """
    model.train()
    runningLoss = 0.0

    for xCatBatch, xNumBatch, yBatch in dataLoader:
        xCatBatch = xCatBatch.to(device)
        xNumBatch = xNumBatch.to(device)
        yBatch = yBatch.to(device)

        optimizer.zero_grad()
        logits = model(xCatBatch, xNumBatch)
        loss = lossFunction(logits, yBatch)
        loss.backward()
        optimizer.step()

        runningLoss += loss.item() * yBatch.size(0)

    return runningLoss / len(dataLoader.dataset)


@torch.no_grad()
def evaluateModel(model, dataLoader, lossFunction, device, threshold: float = 0.5) -> dict:
    """
    Evaluate probability quality, ranking quality, and hard classification.
    """
    model.eval()

    allTargets = []
    allProbabilities = []
    runningLoss = 0.0

    for xCatBatch, xNumBatch, yBatch in dataLoader:
        xCatBatch = xCatBatch.to(device)
        xNumBatch = xNumBatch.to(device)
        yBatch = yBatch.to(device)

        logits = model(xCatBatch, xNumBatch)
        loss = lossFunction(logits, yBatch)
        probabilities = torch.sigmoid(logits)

        runningLoss += loss.item() * yBatch.size(0)
        allTargets.extend(yBatch.cpu().numpy().tolist())
        allProbabilities.extend(probabilities.cpu().numpy().tolist())

    allTargets = np.array(allTargets)
    allProbabilities = np.array(allProbabilities)
    allPredictions = (allProbabilities >= threshold).astype(int)

    metrics = {
        "loss": runningLoss / len(dataLoader.dataset),
        "rocAuc": roc_auc_score(allTargets, allProbabilities),
        "prAuc": average_precision_score(allTargets, allProbabilities),
        "brierScore": brier_score_loss(allTargets, allProbabilities),
        "logLoss": log_loss(allTargets, allProbabilities, labels=[0, 1]),
        "accuracy": accuracy_score(allTargets, allPredictions),
        "precision": precision_score(allTargets, allPredictions, zero_division=0),
        "recall": recall_score(allTargets, allPredictions, zero_division=0),
        "f1": f1_score(allTargets, allPredictions, zero_division=0),
        "threshold": threshold
    }

    return metrics


def printMetrics(splitName: str, epochNumber: int, trainLoss: float, metrics: dict) -> None:
    """
    Print one line of training progress.
    """
    print(
        f"Epoch {epochNumber:02d} | "
        f"Train Loss: {trainLoss:.4f} | "
        f"{splitName} Loss: {metrics['loss']:.4f} | "
        f"ROC-AUC: {metrics['rocAuc']:.4f} | "
        f"PR-AUC: {metrics['prAuc']:.4f} | "
        f"Brier: {metrics['brierScore']:.4f} | "
        f"F1: {metrics['f1']:.4f}"
    )


def runFtTransformerPipeline(csvPath: str) -> None:
    """
    Run the full FT-Transformer workflow and report the final benchmark results.
    """
    setSeed(42)

    df = loadAndCleanData(csvPath)

    print("Cleaned Data Shape:", df.shape)
    print("Missing Values After Cleaning:")
    print(df.isna().sum())
    print()

    dataBundle = prepareDataBundle(df, testSize=0.2, valSize=0.2, randomState=42)

    print("Categorical Columns:", dataBundle.categoricalColumns)
    print("Numerical Columns:", dataBundle.numericalColumns)
    print("Category Cardinalities:", dataBundle.categoryCardinalities)
    print()

    trainDataset = ChurnDataset(
        dataBundle.xTrainCat,
        dataBundle.xTrainNum,
        dataBundle.yTrain
    )
    valDataset = ChurnDataset(
        dataBundle.xValCat,
        dataBundle.xValNum,
        dataBundle.yVal
    )
    testDataset = ChurnDataset(
        dataBundle.xTestCat,
        dataBundle.xTestNum,
        dataBundle.yTest
    )

    trainLoader = DataLoader(trainDataset, batch_size=256, shuffle=True)
    valLoader = DataLoader(valDataset, batch_size=512, shuffle=False)
    testLoader = DataLoader(testDataset, batch_size=512, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print()

    model = FTTransformer(
        categoryCardinalities=dataBundle.categoryCardinalities,
        numNumericalFeatures=len(dataBundle.numericalColumns),
        embeddingDim=32,
        numHeads=4,
        numBlocks=3,
        dropoutRate=0.1
    ).to(device)

    lossFunction = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    numEpochs = 25
    bestValidationPrAuc = -math.inf
    bestStateDict = None

    for epoch in range(1, numEpochs + 1):
        trainLoss = trainOneEpoch(model, trainLoader, optimizer, lossFunction, device)
        validationMetrics = evaluateModel(model, valLoader, lossFunction, device, threshold=0.5)
        printMetrics("Val", epoch, trainLoss, validationMetrics)

        if validationMetrics["prAuc"] > bestValidationPrAuc:
            bestValidationPrAuc = validationMetrics["prAuc"]
            bestStateDict = copy.deepcopy(model.state_dict())

    if bestStateDict is not None:
        model.load_state_dict(bestStateDict)

    finalValidationMetrics = evaluateModel(model, valLoader, lossFunction, device, threshold=0.3)
    finalTestMetrics = evaluateModel(model, testLoader, lossFunction, device, threshold=0.3)

    print("\nBest Validation Results")
    print("-----------------------")
    print(f"ROC-AUC     : {finalValidationMetrics['rocAuc']:.4f}")
    print(f"PR-AUC      : {finalValidationMetrics['prAuc']:.4f}")
    print(f"Brier Score : {finalValidationMetrics['brierScore']:.4f}")
    print(f"Log Loss    : {finalValidationMetrics['logLoss']:.4f}")
    print(f"Accuracy    : {finalValidationMetrics['accuracy']:.4f}")
    print(f"Precision   : {finalValidationMetrics['precision']:.4f}")
    print(f"Recall      : {finalValidationMetrics['recall']:.4f}")
    print(f"F1 Score    : {finalValidationMetrics['f1']:.4f}")

    print("\nFinal Test Benchmark Results")
    print("----------------------------")
    print(f"ROC-AUC     : {finalTestMetrics['rocAuc']:.4f}")
    print(f"PR-AUC      : {finalTestMetrics['prAuc']:.4f}")
    print(f"Brier Score : {finalTestMetrics['brierScore']:.4f}")
    print(f"Log Loss    : {finalTestMetrics['logLoss']:.4f}")
    print(f"Accuracy    : {finalTestMetrics['accuracy']:.4f}")
    print(f"Precision   : {finalTestMetrics['precision']:.4f}")
    print(f"Recall      : {finalTestMetrics['recall']:.4f}")
    print(f"F1 Score    : {finalTestMetrics['f1']:.4f}")
    print(f"Threshold   : {finalTestMetrics['threshold']:.2f}")


if __name__ == "__main__":
    csvPath = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    runFtTransformerPipeline(csvPath)

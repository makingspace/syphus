## Example/test module for the syphus optimization procedure. Implements an
## optimizer of some well-known functions over two floats.
import unittest, random, math, hashes, tables, options, deques, sequtils, strutils
import syphus

# Define concrete state and tabu element types based on the specific domain.
type
  Dim = enum
    x, y
  GenericSolution = array[x..y, float]
  GenericElement = tuple[index: Dim, val: float]
  CellCounter = CountTable[float]
  GenericScoreSolution = ScoreSolution[GenericSolution]
  GenericTabu = Tabu[GenericElement]
  GenericTabuQueue = TabuQueue[GenericElement]

proc `$`*(state: GenericSolution): string = "x: " & $state[x].round(3) & " y: " & $state[y].round(3)

const
  ROUND = 2
  SCALE = 7

type
  OptimizerSettings = object
    xScale, yScale, xMin, xMax, yMin, yMax, xDefault, yDefault: float

# Define callbacks to optimizer function.

proc step(state: GenericSolution, settings: OptimizerSettings): GenericSolution =
  let
    xRange = -settings.xScale..settings.xScale
    xStepSize = random(xRange)
    yRange = -settings.yScale..settings.yScale
    yStepSize = random(yRange)
  var
    newX = state[x] + xStepSize
    newY = state[y] + yStepSize

  if newX < settings.xMin or newX > settings.xMax:
    newX = settings.xDefault
  if newY < settings.yMin or newY > settings.yMax:
    newY = settings.yDefault

  result = [newX, newY]

proc decompose(state: GenericSolution): array[x..y, GenericElement] =
  return [(x, state[x].round(ROUND)), (y, state[y].round(ROUND))]

proc elementIsTabu(tabuQueue: var GenericTabuQueue, element: GenericElement, testScore: Score): bool =
  for tabu in tabuQueue:
    if tabu.value == element:
      return testScore > tabu.aspirationCriteria
  return false

proc isTabu(tabuQueue: var GenericTabuQueue, scoreSolution: GenericScoreSolution): bool =
  let
    decomposed = scoreSolution.solution.decompose()
  result = (
    tabuQueue.elementIsTabu(decomposed[x], scoreSolution.score) or
    tabuQueue.elementIsTabu(decomposed[y], scoreSolution.score)
  )

template runTest(
  xMinIn, xMaxIn, yMinIn, yMaxIn: float,
  objectiveBody: untyped
) {.dirty.} =
  proc objective(state: GenericSolution,
                 diffWith: Option[GenericScoreSolution]): ObjectiveResponse =
    let x = state[x]
    let y = state[y]
    objectiveBody

  let
    xInitial = random(xMinIn..xMaxIn)
    yInitial = random(yMinIn..yMaxIn)

  let initialSolution: GenericSolution = [xInitial, yInitial]
  var settings = OptimizerSettings(
    xDefault: xInitial,
    yDefault: yInitial,
    xMin: xMinIn,
    xMax: xMaxIn,
    yMin: yMinIn,
    yMax: yMaxIn
  )
  settings.xScale = (settings.xMax - settings.xMin) / SCALE
  settings.yScale = (settings.yMax - settings.yMin) / SCALE

  var cellMemory: array[x..y, CellCounter] = [
    initCountTable[float](), initCountTable[float]()
  ]

  proc getNeighborhood(state: GenericSolution): seq[GenericSolution] =
    newSeqWith(20, step(state, settings))

  proc markTabu(tabuQueue: var GenericTabuQueue, scoreSolution: GenericScoreSolution, diffWith: GenericSolution) =
    let decomposed = scoreSolution.solution.decompose()
    for i in x..y:
      let
        delta = abs(diffWith[i] - scoreSolution.solution[i])
        scale = if i == x: settings.xScale else: settings.yScale
        ratio = delta / scale

      proc lifetime(ratio: float): int = max(1, int(100 * ratio))

      let tabu = GenericTabu(
        value: decomposed[i],
        aspirationCriteria: scoreSolution.score,
        lifetime: ratio.lifetime
      )
      tabuQueue.addFirst(tabu)

  proc markGoodMove(scoreSolution: GenericScoreSolution, diffWith: GenericSolution) =
    for i in x..y:
      let rounded = scoreSolution.solution[i].round(ROUND)
      if rounded in cellMemory[i]:
        cellMemory[i].inc(rounded)
      else:
        cellMemory[i][rounded] = 2

  proc restart(tabuQueue: var GenericTabuQueue): GenericSolution =
    for i in x..y:
      cellMemory[i].sort()
      for value in cellMemory[i].keys:
        if not tabuQueue.elementIsTabu((i, value), Inf):
          result[i] = value
          cellMemory[i][value] = 1
          break

  randomize()
  let results = optimize(
    initialSolution,
    getNeighborhood,
    objective,
    proc(_: GenericScoreSolution) = discard,
    isTabu,
    markTabu,
    proc(_: GenericScoreSolution) = discard,
    markGoodMove,
    restart,
    maxSinceMinimum = 40
  )
  echo(
    ("---\n" &
    "starting from $1, $2:\n" &
    "Found optimum: $3 (out of $6 minima)\n" &
    "for x: $4, y: $5 ") % [
    $xInitial.round(4),
    $yInitial.round(4),
    $results.scoreSolution.score,
    $results.scoreSolution.solution[x],
    $results.scoreSolution.solution[y],
    $results.foundMinima
  ])

# xMinIn, xMaxIn yMinIn, yMaxIn
test "mccormick (x=-1.9133)":
 runTest(-1.5, 4.0, -3.0, 4.0):
    result = (sin(x + y) + (x - y).pow(2) - 1.5 * x + 2.5 * y + 1, false)

test "bukin (x=0)":
  runTest(-15.0, -5.0, -3.0, 3.0):
    result = ((y - 0.01 * x.pow(2)).abs.sqrt * 100 + (x + 10).abs + 0.01, false)

test "eggholder (x=-959.6407)":
  runTest(-512.0, 512.0, -512.0, 512.0):
    result = (-(y + 47) * (x / 2.0).abs.sqrt.sin - (x - (y + 47)).abs.sqrt.sin * x, false)

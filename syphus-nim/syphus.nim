## Implementation of the Tabu Search heuristic.
##
## The main API for the Nim version of Syphus is the ``optimize`` proc. It
## takes a series of user-defined callback procs, which is how to
## implement a solver for a specific domain problem.
##
## The procs to implement are:
##  - ``initialSolution: SolutionT``: The starting solution state to be  mutated by the solver.
##  - ``getNeighborhood``: Provide a series of possible next steps from the current solution state.
##  - ``objective``: Evaluate a solution state.
##  - ``registerBestScoreMove``: Callback for handling the step selected from the neighborhood.
##  - ``isTabu``: Determine whether a given solution is currently tabu.
##  - ``markTabu``: Mark a solution state as tabu.
##  - ``markGoodMove``: Callback for handling a non-optimal but desirable step.
##  - ``restart``: Provides a new starting state after some local minimum has been reached.
##
## With these callbacks, ``optimize`` will seek the solution space exposed
## by ``getNeighborhood`` and try to optimize for the lowest cost as returned
## by ``objective``. The other possible callbacks allow the programmer to
## design tabu heuristics and implement supplementary data structures to
## search the space more effectively.
import options, deques, tables, algorithm, sequtils, math

## Generic types are provided so as to create generic signatures for the
## callbacks. These types should be concretized with a ``T`` appropriate for
## the domain of the problem. For instance, in the travelling salesman problem,
## ``T`` might be a sequence of integers representing indices into a sequence
## of stops.
type
  Score* = float ## \
  ## An alias for ``float`` representing the *cost* of a solution under
  ## a specific objective solution, for instance, the total minutes of
  ## driving time in a travelling salesman solution.
  ObjectiveResponse* = tuple[score: Score, stopIteration: bool] ## \
  ## The value returned by the objective function. In addition to the
  ## score, ``stopIteration`` allows for early exit from the optimization
  ## loop if some threshold has been met.
  ScoreSolution*[T] = tuple ## \
    ## A combination of a solution state and that state's score from the
    ## objective function.
    score: Score
    solution: T

  Tabu*[T] = object ## \
    ## An object containing a tabu-active element, its time until expiration,
    ## and the maximum score for it to flag a solution as tabu. An
    ## tabu-active element is an element of a search space (for instance,
    ## the assignment of a vehicle to a route, or the ordering of a single
    ## stop) which is Tabu, meaning that the solution should be passed over
    ## for one that might be less strictly optimal.
    ##
    ## By marking solutions as Tabu (through marking their elements as Tabu)
    ## we can heuristically control the path of the solver through the solution
    ## space and avoid local minima.
    ##
    ## ``value``: The element associated with the tabu solution.
    ##
    ## ``lifetime``: The time that this solution should be tabu.
    ##
    ## ``aspirationCriteria``: The score required to override this tabu value. For instance, this
    ## might be set during ``markTabu`` to the score of the tabu solution, and
    ## then compared with the current score in ``isTabu``, so that solutions
    ## whose scores are sufficiently improved can be selected even if their
    ## elements are tabu.
    value*: T
    lifetime*: int
    aspirationCriteria*: Score
  TabuQueue*[T] = Deque[Tabu[T]] ## \
    ## The queue of all currently tabu elements.

  Response*[T] = object ## \
    ## The type returned by the ``optimize`` function. ``optimize`` will
    ## return the ``ScoreSolution`` that represents the found optimum, in
    ## addition to a count of the local minima found during the search.
    scoreSolution*: ScoreSolution[T]
    foundMinima*: int

proc `$`*(o: ObjectiveResponse): string =
  "(" & $o.score.round(3) & " " & (if o.stopIteration: "STOP" else: "") & ")"

proc cmp(o1, o2: ScoreSolution): int = cmp(o1[0], o2[0])

proc newTabuQueue[T](): TabuQueue[T] =
  initDeque[Tabu[T]]()

# Callback types.

type
  GetNeighborhood*[SolutionT] = proc(state: SolutionT): seq[SolutionT] ## \
  ## Given a solution state, obtain a seq of adjacent solution states for
  ## evaluation.
  Objective*[SolutionT] = proc(state: SolutionT, diffWith: Option[ScoreSolution[SolutionT]]): ObjectiveResponse ## \
  ## Evaluate a solution state with the objective function to be optimized.
  ## Optionally accepts the current optimum for comparison. Return Inf to
  ## represent constraint failure.
  RegisterBestScoreMove*[SolutionT] = proc(scoreSolution: ScoreSolution[SolutionT]) ## \
  ## Optional callback when a solution is selected from a neighborhood.
  IsTabu*[SolutionT, TabuElementT] = proc(tabus: var TabuQueue[TabuElementT], scoreSolution: ScoreSolution[SolutionT]): bool ## \
  ## Use tabu memory structures to evaluate a solution state for tabu-ness.
  MarkTabu*[SolutionT, TabuElementT] = proc(tabus: var TabuQueue[TabuElementT], scoreSolution: ScoreSolution[SolutionT], diffWith: SolutionT) ## \
  ## Mark a given state as tabu, including a more optimal solution for diffing.
  HandleConstraintFailure*[SolutionT] = proc(scoreSolution: ScoreSolution[SolutionT]) ## \
  ## Handle the optional case if some hard constraint is violated.
  MarkGoodMove*[SolutionT] = proc(scoreSolution: ScoreSolution[SolutionT], diffWith: SolutionT) ## \
  ## Mark a solution state for revisting.
  Restart*[SolutionT, TabuElementT] = proc(tabus: var TabuQueue[TabuElementT]): SolutionT ## \
  ## Generate a new initial state from current tabu data.

proc optimize*[SolutionT, TabuElementT](
  initialSolution: SolutionT,
  getNeighborhood: GetNeighborhood[SolutionT],
  objective: Objective[SolutionT],
  registerBestScoreMove: RegisterBestScoreMove[SolutionT],
  isTabu: IsTabu[SolutionT, TabuElementT],
  markTabu: MarkTabu[SolutionT, TabuElementT],
  handleConstraintFailure: HandleConstraintFailure[SolutionT],
  markGoodMove: MarkGoodMove[SolutionT],
  restart: Restart[SolutionT, TabuElementT],

  maxIter = 300,
  maxSinceMinimum = 30,
  maxTabuSize = none(int),
  criteriaReductionMultiplier = 1.1
): Response[SolutionT] =
  type
    ScoreSolutionT = ScoreSolution[SolutionT]
    TabuT = Tabu[TabuElementT]
    ResponseT = Response[SolutionT]
  ## Implements a generic function that takes two types: SolutionT is some
  ## representation of a single solution (or input to the objective function),
  ## TabuElementT is the smallest element of SolutionT that can be marked as
  ## tabu-active. Runs the optimization loop, accepting a series of callbacks
  ## to implement domain-specific logic.

  # Close around SolutionT.
  type StopIterationException = object of Exception
    scoreSolution: ScoreSolutionT

  var tabuQueue = newTabuQueue[TabuElementT]()

  # Closures around tabuQueue.

  proc removeTabu(tabu: TabuT) =
    var buffer = initDeque[TabuT](tabuQueue.len.nextPowerOfTwo)
    # Pop elements from the queue and put them onto a buffer stack until we
    # find the element we want to remove.
    while tabuQueue.len > 0:
      let check = tabuQueue.popLast()
      if check != tabu:
        buffer.addFirst(check)
      else:
        # When we find the desired element, roll back the stack onto the queue.
        while buffer.len > 0:
          tabuQueue.addLast(buffer.popFirst)
        break

  proc manageTabu() =
    ## Count down lifetimes for tabu elements, removing them if expired, and
    ## updating criteria otherwise.
    for tabu in tabuQueue.mitems:
      if tabu.lifetime <= 0:
        removeTabu(tabu)
      else:
        tabu.lifetime -= 1
        tabu.aspirationCriteria *= criteriaReductionMultiplier

    if maxTabuSize.isSome:
      while tabuQueue.len > maxTabuSize.get:
        removeTabu(tabuQueue.peekLast)

  var
    minSolution, currentSolution = initialSolution
    minScore, currentScore = objective(initialSolution, none(ScoreSolutionT)).score

  proc getNextScoreMove(currentScoreSolution: ScoreSolutionT): ScoreSolutionT =
    ## Get the solution neighborhood, and find the optimal non-Tabu solution
    ## from that neighborhood.
    var
      # Get the neighborhood of next possible moves.
      neighborhood = getNeighborhood(currentSolution)
      # Map the objective function over the neighborhood.
      responses = neighborhood.mapIt(objective(it, some(currentScoreSolution)))
      # Initialize a new sequence to hold the score/states.
      scoreSolutions = newSeq[ScoreSolutionT](responses.len)

    for i in 0..responses.high:
      let
        solution = neighborhood[i]
        response = responses[i]
      # Iterate through the response/moves.
      if response.stopIteration:
        # If any of them calls for an early return, package it up and bail out
        # of optimization immediately.
        var earlyReturn = newException(StopIterationException, "Found early return.")
        earlyReturn.scoreSolution = (response.score, solution)
        raise earlyReturn
      else:
        # Otherwise, add the score/solution minus the flag.
        scoreSolutions[i] = (response.score, solution)

    # Sort the resulting score/states, optimal first.
    scoreSolutions.sort do (x, y: ScoreSolutionT) -> int: cmp(x.score, y.score)

    # Iterate through the neighborhood, looking for the first valid score/state.
    var foundBestMove = false
    for scoreSolution in scoreSolutions:
      if not foundBestMove and not tabuQueue.isTabu(scoreSolution):
        # Choose the best (first) non-tabu move.
        result = scoreSolution
        foundBestMove = true
      if scoreSolution.score < currentScoreSolution.score:
        # Non-optimal moves that are an improvement on the current minimum,
        # mark to revisit.
        markGoodMove(scoreSolution, currentScoreSolution.solution)

    if not foundBestMove:
      # If all moves are tabu, fall back to the optimal tabu move.
      result = scoreSolutions[0]

  # Main optimization loop.
  var iterCount, iterSinceLastMinimum, globalMinimaFound = 0

  while iterCount < maxIter:
    iterCount += 1
    iterSinceLastMinimum += 1
    let currentScoreSolution = (currentScore, currentSolution)
    var nextScoreMove: ScoreSolutionT

    # Get next step from current state.
    try:
      nextScoreMove = getNextScoreMove(currentScoreSolution)
    except StopIterationException as e:
      # Early-exit handling.
      return ResponseT(
        scoreSolution: e.scoreSolution,
        foundMinima: globalMinimaFound
      )
    # We're not exiting early, so let's handle the next step.
    registerBestScoreMove(nextScoreMove)
    # Mark tabu wherever we came from.
    tabuQueue.markTabu(currentScoreSolution, nextScoreMove.solution)
    if nextScoreMove.score == Inf:
      handleConstraintFailure(nextScoreMove)
    elif nextScoreMove.score < minScore:
      # Hillclimbing: handle a new minimum.
      iterSinceLastMinimum = 0
      globalMinimaFound += 1
      minScore = nextScoreMove.score
      minSolution = nextScoreMove.solution
    # Prepare for next iteration.
    if iterSinceLastMinimum >= maxSinceMinimum:
      # If we have reached a local minimum, attempt to restart.
      currentSolution = restart(tabuQueue)
      currentScore = objective(currentSolution, none(ScoreSolutionT)).score
      iterSinceLastMinimum = 0
    else:
      # Otherwise, move to the next position in the space.
      currentSolution = nextScoreMove.solution
      currentScore = nextScoreMove.score
    # Manage tabu data.
    manageTabu()

  result = ResponseT(
    scoreSolution: (minScore, minSolution),
    foundMinima: globalMinimaFound
  )

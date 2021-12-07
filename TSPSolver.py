#!/usr/bin/python3
import copy

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
import copy


def k_opt_swap(tsp, splits):

	k = len(splits)
	perm = [x for x in itertools.product([True, False], repeat=k + 1)]
	pieces = []
	cities = tsp.route
	bs = TSPSolution(copy.deepcopy(cities))

	for i in range(k + 1):
		if i == 0:
			b = splits[i]
			pieces.append(cities[:b])
		elif i == k:
			a = splits[i - 1]
			pieces.append(cities[a:])
		else:
			a = splits[i - 1]
			b = splits[i]
			pieces.append(cities[a:b])

	for i in range(len(perm)):
		soln = copy.deepcopy(pieces)
		for j in range(len(perm[i])):
			if perm[i][j]:
				soln[j].reverse()
		soln = [x for y in soln for x in y]
		tsp = TSPSolution(soln)
		if tsp.cost < bs.cost:
			bs = tsp

	if bs == cities:
		return None
	else:
		return bs


class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def greedy( self,time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		beginCity = 0
		count = 0
		bssf = TSPSolution(cities)
		bssf.cost = math.inf
		start_time = time.time()
		# Keep searching while a tour is not found, greedy options are not exhausted, and time limit has not been exceeded
		# Time and Space Complexity: O(n^3)
		while beginCity < ncities and time.time()-start_time < time_allowance:
			# print("Starting city: " + str(beginCity))
			route = [cities[beginCity]]
			# Add a city to the route until all cities have been visited
			for i in range(ncities-1):
				minCost = math.inf
				nextCity = None
				# Find the route with the cheapest cost to a city that has not been visited yet
				for j in range(ncities):
					if cities[j] not in route:
						costTo = route[i].costTo(cities[j])
						if costTo < math.inf and costTo < minCost:
							# Found a path cheaper than any other path found yet
							minCost = costTo
							nextCity = cities[j]
				if nextCity is not None:
					route.append(nextCity)
				else:
					# No available city, try another tour
					break
				pass
			if len(route) == ncities:
				# Every city was visited, possible valid tour
				pvt = TSPSolution(route)
				count += 1	
				if pvt.cost < bssf.cost:
					# Found a valid route better than previous solutions
					foundTour = True
					bssf = pvt
			# Check next city for a greedy trial
			beginCity += 1
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results
	
	
	
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''
		
	def branchAndBound( self, time_allowance=60.0 ):
		pass



	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
	def fancy( self,time_allowance=60.0 ):
		pass



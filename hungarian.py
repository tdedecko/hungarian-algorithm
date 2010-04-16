#!/usr/bin/python
"""
Implementation of the Hungarian (Munkres) Algorithm using Python and NumPy
References: http://www.ams.jhu.edu/~castello/362/Handouts/hungarian.pdf
	    http://weber.ucsd.edu/~vcrawfor/hungar.pdf
	    http://en.wikipedia.org/wiki/Hungarian_algorithm
	    http://www.public.iastate.edu/~ddoty/HungarianAlgorithm.html
	    http://www.clapper.org/software/python/munkres/
"""

# Module Information.
__version__   = "1.0.1"
__author__    = "Thom Dedecko"
__url__       = "http://github.com/tdedecko/hungarian-algorithm"
__copyright__ = "(c) 2010 Thom Dedecko"
__license__   = "MIT License"


class HungarianError(Exception): pass


# Import numpy. Error if fails
try:
	import numpy
except ImportError:
	raise HungarianError( "NumPy is not installed." )


class Hungarian:
	"""
	Implementation of the Hungarian (Munkres) Algorithm using NumPy.
	
	Usage:
		hungarian = Hungarian(costMatrix)
		hungarian.calculate()
	or
		hungarian = Hungarian()
		hungarian.calculate(costMatrix)

	Handle Profit matrix:
		hungarian = Hungarian(profitMatrix, isProfitMatrix=True)
	or
		costMatrix = Hungarian.makeCostMatrix(profitMatrix)

	The matrix will be automatically padded if it is not square.
	The matrix can be padded with:
		paddedMatrix = Hungarian.padMatrix(costMatrix)

	Get results and total potential after calculation:
		hungarian.getResults()
		hungarian.getTotalPotential()
	"""

	
	def __init__(self, inputMatrix=None, isProfitMatrix=False):
		"""
		inputMatrix is a List of Lists.
		inputMatrix is assumed to be a cost matrix unless isProfitMatrix is True.
		"""
		if not (inputMatrix == None):
			# Save input
			self._inputMatrix = numpy.array(inputMatrix)
			self._maxColumn = len(inputMatrix[0])
			self._maxRow = len(inputMatrix)

			# Pad matrix if necessary
			paddedMatrix = self.padMatrix(inputMatrix)
			myMatrix = numpy.array(paddedMatrix)

			# Convert matrix to profit matrix if necessary
			if isProfitMatrix:
				myMatrix = self.makeCostMatrix(myMatrix)

			self._costMatrix = myMatrix
			self._size = len(myMatrix)
			self._shape = myMatrix.shape

			# Results from algorithm.
			self._results = []
			self._totalPotential = 0
		else:
			self._costMatrix = None


	def getResults(self):
		"""Get results after calculation."""
		return self._results


	def getTotalPotential(self):
		"""Returns expected value after calculation."""
		return self._totalPotential


	def calculate(self, inputMatrix=None, isProfitMatrix=False):
		"""
		Implementation of the Hungarian (Munkres) Algorithm.

		inputMatrix is a List of Lists.
		inputMatrix is assumed to be a cost matrix unless isProfitMatrix is True.
		"""
		# Handle invalid and new matrix inputs.
		if (inputMatrix == None) and (self._costMatrix == None):
			raise HungarianError( "Invalid input" )
		elif not (inputMatrix == None):
			self.__init__(inputMatrix, isProfitMatrix)
		
		resultMatrix = self._costMatrix.copy()

		# Step 1: Subtract row mins from each row.
		for index, row in enumerate(resultMatrix):
			resultMatrix[index] -= row.min()	

		# Step 2: Subtract column mins from each column.
		for index, column in enumerate(resultMatrix.transpose()):
			resultMatrix[:,index] -= column.min()
	
		# Step 3: Use minimum number of lines to cover all zeros in the matrix.
		# If the total covered rows+columns is not equal to the matrix size then adjust matrix and repeat.
		totalCovered = 0
		while totalCovered < self._size:
			# Find minimum number of lines to cover all zeros in the matrix and find total covered rows and columns.
			coverZeros = CoverZeros(resultMatrix)
			coveredRows = coverZeros.getCoveredRows()
			coveredColumns = coverZeros.getCoveredColumns()
			totalCovered = len(coveredRows) + len(coveredColumns)
			
			# if the total covered rows+columns is not equal to the matrix size then adjust matrix by minimum uncovered num (m).
			if totalCovered < self._size:
				resultMatrix = self.__adjustMatrixByMinUncoveredNum(resultMatrix, coveredRows, coveredColumns)

		# Step 4: Starting with the top row, work your way downwards as you make assignments.
		# Find single zeros in rows or columns. Add them to final result and remove them and their associated row/column from the matrix.
		expectedResults = min( self._maxColumn, self._maxRow )
		zeroLocations = (resultMatrix == 0)
		while len(self._results) != expectedResults:

			# If number of zeros in the matrix is zero before finding all the results then an error has occured.
			if not zeroLocations.any():
				raise HungarianError( "Unable to find results. Algorithm has failed." )			

			matchedRows = numpy.array([])
			matchedColumns = numpy.array([])
			
			# Find results and mark rows and columns for deletion
			matchedRows, matchedColumns = self.__findMatches(zeroLocations)

			# Make arbitrary selection
			totalMatched = len(matchedRows) + len(matchedColumns)
			if totalMatched == 0:
				matchedRows, matchedColumns = self.__selectArbitraryMatch(zeroLocations)
			
			# Delete rows and columns
			for row in matchedRows:
				zeroLocations[row] = False
			for column in matchedColumns:
				zeroLocations[:,column] = False

			# Save Results
			self.__setResults(zip(matchedRows, matchedColumns))
	
		# Calculate total potential
		value = 0
		for row, column in self._results:
			value += self._inputMatrix[row,column]
		self._totalPotential = value


	def makeCostMatrix(self, profitMatrix):
		"""
		Converts a profit matrix into a cost matrix.
		Expects NumPy objects as input.
		"""
		# subtract profit matrix from a matrix made of the max value of the profit matrix 
		matrixShape = profitMatrix.shape
		offsetMatrix = numpy.ones(matrixShape) * profitMatrix.max()
		costMatrix = offsetMatrix - profitMatrix
		return costMatrix

	
	def padMatrix(self, myMatrix):
		"""
		If matrix is not square, then make it square by padding the matrix with zeros.
		Method expects input of a list of lists and not NumPy objects.
		"""
		# Get matrix dimensions
		numCols = len(myMatrix[0])
		numRows = len(myMatrix)
		matrixSize = max(numCols, numRows)
		
		paddedMatrix = []

		# Pad columns with zeros
		for index, row in enumerate(myMatrix):
			numMissingCols = matrixSize - len(row)
			if numMissingCols < 0:
				raise HungarianError( "Matrix has variable row length" )
			
			# Update paddedMatrix
			newRow = myMatrix[index] + ([0] * numMissingCols)
			paddedMatrix.append(newRow)
		
		# Pad rows with zeros
		numMissingRows = matrixSize - len(myMatrix)
		for missingRow in range(numMissingRows):
			paddedMatrix.append([0] * matrixSize)
		
		return paddedMatrix


	def __adjustMatrixByMinUncoveredNum(self, resultMatrix, coveredRows, coveredColumns):
		"""Subtract m from every uncovered number and add m to every element covered with two lines."""
		# Calculate minimum uncovered number (m)
		elements = []
		for rowIndex, row in enumerate(resultMatrix):
			if not rowIndex in coveredRows:
				for index, element in enumerate(row):
					if not index in coveredColumns:
						elements.append(element)
		minUncoveredNum = min(elements)
		
		# Add m to every covered element
		adjustedMatrix = resultMatrix
		for row in coveredRows:
			adjustedMatrix[row] += minUncoveredNum
		for column in coveredColumns:
			adjustedMatrix[:,column] += minUncoveredNum
		
		# Subtract m from every element
		mMatrix = numpy.ones(self._shape) * minUncoveredNum
		adjustedMatrix -= mMatrix

		return adjustedMatrix


	def __findMatches(self, zeroLocations):
		"""Returns rows and columns with matches in them."""
		markedRows = numpy.array([])
		markedColumns = numpy.array([])

		# Mark rows and columns with matches
		# Iterate over rows
		for index, row in enumerate(zeroLocations):
			rowIndex = numpy.array([index])
			if (numpy.sum(row) == 1):
				columnIndex, = numpy.where(row)
				markedRows, markedColumns = self.__markRowsAndColumns(markedRows, markedColumns, rowIndex, columnIndex)

		# Iterate over columns
		for index, column in enumerate(zeroLocations.transpose()):
			columnIndex = numpy.array([index])
			if (numpy.sum(column) == 1):
				rowIndex, = numpy.where(column)
				markedRows, markedColumns = self.__markRowsAndColumns(markedRows, markedColumns, rowIndex, columnIndex)

		return (markedRows, markedColumns)


	def __markRowsAndColumns(self, markedRows, markedColumns, rowIndex, columnIndex):
		"""Check if column or row is marked. If not marked then mark it."""
		newMarkedRows = markedRows
		newMarkedColumns = markedColumns
		if ( not (markedRows == rowIndex).any() ) and ( not (markedColumns == columnIndex).any() ):
			newMarkedRows = numpy.insert(markedRows, len(markedRows), rowIndex)
			newMarkedColumns = numpy.insert(markedColumns, len(markedColumns), columnIndex)
		return (newMarkedRows, newMarkedColumns)


	def __selectArbitraryMatch(self, zeroLocations):
		"""Selects row column combination with minimum number of zeros in it."""
		# Count number of zeros in row and column combinations
		rowList, columnList = numpy.where(zeroLocations)
		zeroCount = []
		for index, row in enumerate(rowList):
			totalZeros = numpy.sum(zeroLocations[row]) + numpy.sum(zeroLocations[:,columnList[index]])
			zeroCount.append( totalZeros )

		# Get the row column combination with the minimum number of zeros.
		listIndex = zeroCount.index( min(zeroCount) )
		row = numpy.array([ rowList[listIndex] ])
		column = numpy.array([ columnList[listIndex] ])

		return (row, column)
		
		
	def __setResults(self, resultList):
		"""Set results during calculation."""
		# Check if results values are out of bound from input matrix (because of matrix being padded).
		# Add results to results list.
		for result in resultList:
			row, column = result
			if (row < self._maxRow) and (column < self._maxColumn):
				newResult = ( int(row), int(column) )
				self._results.append(newResult)

	
class CoverZeros:
	"""
	Use minimum number of lines to cover all zeros in the matrix.
	Algorithm based on: http://weber.ucsd.edu/~vcrawfor/hungar.pdf
	"""

	
	def __init__(self, matrix):
		"""
		Input a matrix and save it as a boolean matrix to designate zero locations.
		Run calculation procedure to generate results.
		"""
		# Find zeros in matrix
		self._zeroLocations = (matrix == 0)
		self._shape = matrix.shape
		
		# Choices starts without any choices made.
		self._choices = numpy.zeros(self._shape, dtype=bool)

		self._markedRows = []
		self._markedColumns = []

		# marks rows and columns
		self.__calculate()

		# Draw lines through all unmarked rows and all marked columns.
		self._coveredRows = list( set( range(self._shape[0]) ) - set( self._markedRows ) )
		self._coveredColumns = self._markedColumns


	def getCoveredRows(self):
		"""Return list of covered rows."""
		return self._coveredRows

	
	def getCoveredColumns(self):
		"""Return list of covered columns."""
		return self._coveredColumns


	def __calculate(self):
		"""
		Calculates minimum number of lines necessary to cover all zeros in a matrix.
		Algorithm based on: http://weber.ucsd.edu/~vcrawfor/hungar.pdf
		"""
		while True:
			# Erase all marks.
			self._markedRows = []
			self._markedColumns = []
			
			# Mark all rows in which no choice has been made.
			for index, row in enumerate(self._choices):
				if not row.any():
					self._markedRows.append(index)

			# If no marked rows then finish.
			if self._markedRows == []:
				return True

			# Mark all columns not already marked which have zeros in marked rows.
			numMarkedColumns = self.__markNewColumnsWithZerosInMarkedRows()

			# If no new marked columns then finish.
			if numMarkedColumns == 0:
				return True

			# While there is some choice in every marked column.
			while self.__choiceInAllMarkedColumns():
				# Some Choice in every marked column.

				# Mark all rows not already marked which have choices in marked columns.
				numMarkedRows = self.__markNewRowsWithChoicesInMarkedColumns()

				# If no new marks then Finish.
				if numMarkedRows == 0:
					return True

				# Mark all columns not already marked which have zeros in marked rows.
				numMarkedColumns = self.__markNewColumnsWithZerosInMarkedRows()

				# If no new marked columns then finish.
				if numMarkedColumns == 0:
					return True

			# No choice in one or more marked columns.
			# Find a marked column that does not have a choice.
			choiceColumnIndex = self.__findMarkedColumnWithoutChoice()

			while choiceColumnIndex != None:
				# Find a zero in the column indexed that does not have a row with a choice.
				choiceRowIndex = self.__findRowWithoutChoice(choiceColumnIndex)

				# Check if an available row was found.
				newChoiceColumnIndex = None
				if choiceRowIndex == None:
					# Find a good row to accomodate swap. Find its column pair.
					choiceRowIndex, newChoiceColumnIndex = self.__findBestChoiceRowAndNewColumn(choiceColumnIndex)

					# Delete old choice.
					self._choices[choiceRowIndex, newChoiceColumnIndex] = False
						
				# Set zero to choice.
				self._choices[choiceRowIndex,choiceColumnIndex] = True

				# Loop again if choice is added to a row with a choice already in it.
				choiceColumnIndex = newChoiceColumnIndex


	def __markNewColumnsWithZerosInMarkedRows(self):
		"""Mark all columns not already marked which have zeros in marked rows."""
		numMarkedColumns = 0
		for index, column in enumerate(self._zeroLocations.transpose()):
			if not index in self._markedColumns:
				if column.any():
					rowIndexList, = numpy.where(column)
					zerosInMarkedRows = ( set(self._markedRows) & set(rowIndexList) ) != set([])
					if zerosInMarkedRows:
						self._markedColumns.append(index)
						numMarkedColumns += 1
		return numMarkedColumns

	
	def __markNewRowsWithChoicesInMarkedColumns(self):
		"""Mark all rows not already marked which have choices in marked columns."""
		numMarkedRows = 0
		for index, row in enumerate(self._choices):
			if not index in self._markedRows:
				if row.any():
					columnIndex, = numpy.where(row)
					if columnIndex in self._markedColumns:
						self._markedRows.append(index)
						numMarkedRows += 1
		return numMarkedRows

	
	def __choiceInAllMarkedColumns(self):
		"""Return Boolean True if there is a choice in all marked columns. Returns boolean False otherwise."""
		for columnIndex in self._markedColumns:
			if not self._choices[:,columnIndex].any():
				return False
		return True


	def __findMarkedColumnWithoutChoice(self):
		"""Find a marked column that does not have a choice."""		
		for columnIndex in self._markedColumns:
			if not self._choices[:,columnIndex].any():
				return columnIndex

		raise HungarianError( "Could not find a column without a choice. Failed to cover matrix zeros. Algorithm has failed." )

		
	def __findRowWithoutChoice(self, choiceColumnIndex):
		"""Find a row without a choice in it for the column indexed. If a row does not exist then return None."""
		rowIndexList, = numpy.where(self._zeroLocations[:,choiceColumnIndex])
		for rowIndex in rowIndexList:
			if not self._choices[rowIndex].any():
				return rowIndex
		
		# All rows have choices. Return None.
		return None


	def __findBestChoiceRowAndNewColumn(self, choiceColumnIndex):
		"""
		Find a row index to use for the choice so that the column that needs to be changed is optimal.
		Return a random row and column if unable to find an optimal selection.
		"""
		rowIndexList, = numpy.where(self._zeroLocations[:,choiceColumnIndex])
		for rowIndex in rowIndexList:
			columnIndexList, = numpy.where(self._choices[rowIndex])
			columnIndex = columnIndexList[0]
			if self.__findRowWithoutChoice(columnIndex) != None:
				return (rowIndex, columnIndex)

		# Cannot find optimal row and column. Return a random row and column.
		from random import shuffle
		shuffle(rowIndexList)
		columnIndex, = numpy.where(self._choices[rowIndexList[0]])
		return (rowIndexList[0], columnIndex[0])


if __name__ == '__main__':
	profitMatrix = [
		[62,75,80,93,95,97],
		[75,80,82,85,71,97],
		[80,75,81,98,90,97],
		[78,82,84,80,50,98],
		[90,85,85,80,85,99],
		[65,75,80,75,68,96]]

	hungarian = Hungarian(profitMatrix, isProfitMatrix=True)
	hungarian.calculate()
	print "Expected value:\t\t543"
	print "Calculated value:\t", hungarian.getTotalPotential() # = 543
	print "Expected results:\n\t[(0, 4), (2, 3), (5, 5), (4, 0), (1, 1), (3, 2)]"
	print "Results:\n\t", hungarian.getResults()
	print "-" * 80

	costMatrix = [
		[4,2,8],
		[4,3,7],
		[3,1,6]]
	hungarian = Hungarian(costMatrix)
	hungarian.calculate()
	print "Expected value:\t\t12"
	print "Calculated value:\t", hungarian.getTotalPotential() # = 12
	print "Expected results:\n\t[(0, 1), (1, 0), (2, 2)]"
	print "Results:\n\t", hungarian.getResults()
	print "-" * 80

	profitMatrix = [
		[62,75,80,93,0,97],
		[75,0,82,85,71,97],
		[80,75,81,0,90,97],
		[78,82,0,80,50,98],
		[0,85,85,80,85,99],
		[65,75,80,75,68,0]]
	hungarian = Hungarian()
	hungarian.calculate(profitMatrix, isProfitMatrix=True)
	print "Expected value:\t\t523"
	print "Calculated value:\t", hungarian.getTotalPotential() # = 523
	print "Expected results:\n\t[(0, 3), (2, 4), (3, 0), (5, 2), (1, 5), (4, 1)]"
	print "Results:\n\t", hungarian.getResults()
	print "-" * 80

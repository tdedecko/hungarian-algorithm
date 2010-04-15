Implementation of the Hungarian (Munkres) Algorithm using Python and NumPy.

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


Released under MIT License.
Source repository: git://github.com/tdedecko/hungarian-algorithm.git

References: 
	http://www.ams.jhu.edu/~castello/362/Handouts/hungarian.pdf
	http://weber.ucsd.edu/~vcrawfor/hungar.pdf
	http://en.wikipedia.org/wiki/Hungarian_algorithm
	http://www.public.iastate.edu/~ddoty/HungarianAlgorithm.html
	http://www.clapper.org/software/python/munkres/

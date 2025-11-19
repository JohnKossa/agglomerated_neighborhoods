# Algorithm

Shatter the universe, and glue it back together

1. Start with a map file (parquet) for all parcels and a map file for all tiles
   1. Parcels map will contain
      1. key
      2. built_area_sqft (not initally present everywhere, backfilled with spatial lag later in the algorithm)
      3. land_area_sqft
      4. adj_sale_price (optional)
      5. assessed_value
      6. geometry
   2. Tiles will contain
      1. key
      2. r_squared (initially empty, but filled after each join)
2. Spatial lag algorithm on built square footage to ensure that all parcels have it (this infills vacant parcels).
   1. Spatial lag is the 3 nearest neighbors, weighted by inverse of distance to the parcel we're filling
3. Tag each built parcel with its market value proxy (average of adjusted sale price and assessor value if both exist, just assessor value if no sale exists)
4. Spatial lag the market value proxy to any fields that don't have it (this infills vacant parcels as the average of nearby built parcels).
   1. Spatial lag is the 3 nearest neighbors, weighted by inverse of distance to the parcel we're filling
5. Decide on a desired ending number of tiles (user input)
6. For each prospective adjacent tile pair, OLS all contained parcels with built size and land size predicting against market value proxy, record the R^2
   1. Adjacent tiles will be decided based on adjacency or intersections with a 30 foot buffer
   2. Use "rook rule" for intersection, meaning that they must either share a boundary or intersect. A single point of tangency isn't sufficient
   3. If a prospective join doesn't have at least 3 sales, set the r_squared to 0
7. Find the pair with the best prospective join based on the R^2 and dissolve them into a single tile.
   1. In the unlikely event of an R squared tie, choose the tile with the highest parcel count
8. Recompute the prospective joins between the recently joined tile and all its neighbors
9. Output an intermediate parquet file of the tiles and their R^2 values
   1. Should contain
      1. key
      2. geometry
      3. r_squared (recorded from the prospective join that produced the tile)
   2. Should be named `intermediate_tiles_<iteration_num>.parquet` 
10. Repeat from step 6 until either no joins remain or we've reached the desired ending number of tiles

# Speed Concerns

This algorithm will likely have a very expensive runtime.
It cannot be parallelized because each join changes the global state, so sharding the process changes the behavior.

As such, we should avoid redoing any computations that have not changed since the last iteration.
We should also memo-ize fairly aggressively.

We should probably maintain a list of neighbor relations, and after the initial neighbor calculation only recompute neighbors if one of our neighbors has changed.

The r_squared value between adjacent neighbors should similarly not change unless one of the tiles in that prospective join has changed (by being joined).

We may need to maintain a working object of prospective_joins to hold the memoization.

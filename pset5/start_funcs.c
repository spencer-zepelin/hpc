
// John's
void parallel_randomizeBodies(Body * bodies, int nBodies, int nBodies_per_rank, int mype, int nprocs)
{
	uint64_t seed = 42;

	// velocity scaling term
	double vm = 1.0e-2;
#ifdef OPENMP
#pragma omp parallel for shared(bodies) schedule(static)
#endif
	for (int i = 0; i < nBodies_per_rank; i++)
	{
		int global_particle_id = (mype * nBodies_per_rank) + i;
		// Fast forward seed to this particle's location in the global PRNG stream.
		// We forward 7 x particle_id, as each particle requires 7 PRNG samples.
		uint64_t particle_seed = fast_forward_LCG(seed, global_particle_id * 7);

		// Initialize positions
		bodies[i].x =  2.0 * LCG_random_double(&particle_seed) - 1.0;
		bodies[i].y =  2.0 * LCG_random_double(&particle_seed) - 1.0;
		bodies[i].z = (2.0 * LCG_random_double(&particle_seed) - 1.0) * 0.1;

		// Intialize velocities
		bodies[i].vx =  2.0 * vm * LCG_random_double(&particle_seed) - vm;
		bodies[i].vy =  2.0 * vm * LCG_random_double(&particle_seed) - vm;
		bodies[i].vz = (2.0 * vm * LCG_random_double(&particle_seed) - vm) * 0.1;

		// Give it a spin
		if( bodies[i].x > 0 )
			bodies[i].vy =  5*fabs(bodies[i].vy);
		else
			bodies[i].vy = -5*fabs(bodies[i].vy);

		// Initialize masses so that total mass of system is constant
		// regardless of how many bodies are simulated.
		bodies[i].mass = LCG_random_double(&particle_seed) / nBodies;
	}
}

// v is position
void parallel_randomizeBodies(Body * bodies, int nBodies, int nBodies_per_rank, int mype, int nprocs)
{
	uint64_t seed = 42;

	// velocity scaling term
	double vm = 1.0e-2;
#ifdef OPENMP
#pragma omp parallel for shared(bodies) schedule(static)
#endif
	for (int i = 0; i < nBodies_per_rank; i++)
	{
		int global_particle_id = (mype * nBodies_per_rank) + i;
		// Fast forward seed to this particle's location in the global PRNG stream.
		// We forward 7 x particle_id, as each particle requires 7 PRNG samples.
		uint64_t particle_seed = fast_forward_LCG(seed, global_particle_id * 4);

		// Initialize positions
		double d1 = LCG_random_double(&particle_seed);
		double d2 = LCG_random_double(&particle_seed);
		double d3 = LCG_random_double(&particle_seed);

		bodies[i].x =  2.0 * d1 - 1.0;
		bodies[i].y =  2.0 * d2 - 1.0;
		bodies[i].z = (2.0 * d3 - 1.0) * 0.1;

		// Intialize velocities
		bodies[i].vx =  2.0 * vm * d1 - vm;
		bodies[i].vy =  2.0 * vm * d2 - vm;
		bodies[i].vz = (2.0 * vm * d3 - vm) * 0.1;


		// Initialize masses so that total mass of system is constant
		// regardless of how many bodies are simulated.
		bodies[i].mass = LCG_random_double(&particle_seed) / nBodies;
	}
}
// v is -position
void parallel_randomizeBodies(Body * bodies, int nBodies, int nBodies_per_rank, int mype, int nprocs)
{
	uint64_t seed = 42;

	// velocity scaling term
	double vm = 1.0e-2;
#ifdef OPENMP
#pragma omp parallel for shared(bodies) schedule(static)
#endif
	for (int i = 0; i < nBodies_per_rank; i++)
	{
		int global_particle_id = (mype * nBodies_per_rank) + i;
		// Fast forward seed to this particle's location in the global PRNG stream.
		// We forward 7 x particle_id, as each particle requires 7 PRNG samples.
		uint64_t particle_seed = fast_forward_LCG(seed, global_particle_id * 4);

		// Initialize positions
		double d1 = LCG_random_double(&particle_seed);
		double d2 = LCG_random_double(&particle_seed);
		double d3 = LCG_random_double(&particle_seed);

		bodies[i].x =  2.0 * d1 - 1.0;
		bodies[i].y =  2.0 * d2 - 1.0;
		bodies[i].z = (2.0 * d3 - 1.0) * 0.1;

		// Intialize velocities
		bodies[i].vx =  -(2.0 * vm * d1 - vm);
		bodies[i].vy =  -(2.0 * vm * d2 - vm);
		bodies[i].vz = -(2.0 * vm * d3 - vm) * 0.1;


		// Initialize masses so that total mass of system is constant
		// regardless of how many bodies are simulated.
		bodies[i].mass = LCG_random_double(&particle_seed) / nBodies;
	}
}
// v is -position
void parallel_randomizeBodies(Body * bodies, int nBodies, int nBodies_per_rank, int mype, int nprocs)
{
	uint64_t seed = 42;

	// velocity scaling term
	double vm = 1.0e-2;
#ifdef OPENMP
#pragma omp parallel for shared(bodies) schedule(static)
#endif
	for (int i = 0; i < nBodies_per_rank; i++)
	{
		int global_particle_id = (mype * nBodies_per_rank) + i;
		// Fast forward seed to this particle's location in the global PRNG stream.
		// We forward 7 x particle_id, as each particle requires 7 PRNG samples.
		uint64_t particle_seed = fast_forward_LCG(seed, global_particle_id * 4);

		// Initialize positions
		double d1 = LCG_random_double(&particle_seed);
		double d2 = LCG_random_double(&particle_seed);
		double d3 = LCG_random_double(&particle_seed);

		bodies[i].x =  2.0 * d1 - 1.0;
		bodies[i].y =  2.0 * d2 - 1.0;
		bodies[i].z = (2.0 * d3 - 1.0) * 0.1;

		// Intialize velocities
		bodies[i].vx =  -(2.0 * vm * d1 - vm);
		bodies[i].vy =  -(2.0 * vm * d2 - vm);
		bodies[i].vz = -(2.0 * vm * d3 - vm) * 0.1;


		// Initialize masses so that total mass of system is constant
		// regardless of how many bodies are simulated.
		bodies[i].mass = LCG_random_double(&particle_seed) / nBodies;
	}
}

// v spin
void parallel_randomizeBodies(Body * bodies, int nBodies, int nBodies_per_rank, int mype, int nprocs)
{
	uint64_t seed = 42;

	// velocity scaling term
	double vm = 1.0e-2;
#ifdef OPENMP
#pragma omp parallel for shared(bodies) schedule(static)
#endif
	for (int i = 0; i < nBodies_per_rank; i++)
	{
		int global_particle_id = (mype * nBodies_per_rank) + i;
		// Fast forward seed to this particle's location in the global PRNG stream.
		// We forward 7 x particle_id, as each particle requires 7 PRNG samples.
		uint64_t particle_seed = fast_forward_LCG(seed, global_particle_id * 4);

		// Initialize positions
		double d1 = LCG_random_double(&particle_seed);
		double d2 = LCG_random_double(&particle_seed);
		double d3 = LCG_random_double(&particle_seed);

		bodies[i].x =  2.0 * d1 - 1.0;
		bodies[i].y =  2.0 * d2 - 1.0;
		bodies[i].z = (2.0 * d3 - 1.0) * 0.1;

		// Intialize velocities
		bodies[i].vx =  -(2.0 * vm * d1 - vm);
		bodies[i].vy =  -(2.0 * vm * d2 - vm);
		bodies[i].vz = -(2.0 * vm * d3 - vm) * 0.1;

		if( bodies[i].x > 0 ){
			bodies[i].vy =  2*fabs(bodies[i].vy);
		} else {
			bodies[i].vy =  -2*fabs(bodies[i].vy);
		}
		if( bodies[i].y > 0 ){
			bodies[i].vx =  -2*fabs(bodies[i].vx);
		} else {
			bodies[i].vx =  2*fabs(bodies[i].vx);
		}

		// Initialize masses so that total mass of system is constant
		// regardless of how many bodies are simulated.
		bodies[i].mass = LCG_random_double(&particle_seed) / nBodies;
	}
}



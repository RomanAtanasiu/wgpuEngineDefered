#pragma once

#include "includes.h"
#include <vector>
#include "glm/glm.hpp"

typedef enum { HALTON,
	GRID,
	CIRCULAR,
	CIRCULAR_HELIX,
	UNIFORM,
	ROTATING_GRID,
	ROTATING_CIRCULAR } SamplerType;

std::vector<glm::vec2> generate_halton_2x3(int num_samples);
std::vector<glm::vec2> generate_grid(int samples);

/*generate random samples in unifrom distribution in [0,1], max samples 32*/
std::vector<glm::vec2> generate_unifrom(int samples);
void rotate_samples(std::vector<glm::vec2> &samples, float angle_radians);

/*  generates samples in a circle, divides into two halfes, it alternates between halfs to assign in order
* max radius 0.5*/
std::vector<glm::vec2> generate_circular_helix(int samples, float radius);

/* generates samples in a circle, max radius 0.5
 */
std::vector<glm::vec2> generate_circular(int samples, float radius);

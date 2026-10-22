#include "utils.h"
#include <random>
#include "sampler.h"
#include "spdlog/spdlog.h"

float radical_inverse(int n, int base) {
	float result = 0.0f;
	float fraction = 1.0f / base;

	while (n > 0) {
		int digit = n % base;
		result += digit * fraction;
		n /= base;
		fraction /= base;
	}

	return result;
}


std::vector<glm::vec2> generate_halton_2x3(int num_samples) {
	std::vector<glm::vec2> samples;
	samples.reserve(num_samples);

	for (int i = 1; i <= num_samples; ++i) {
		float x = radical_inverse(i, 2);
		float y = radical_inverse(i, 3); 

		samples.emplace_back(x, y);
	}

	return samples;
}


std::vector<glm::vec2> generate_grid(int samples) {
	std::vector<glm::vec2> points;

	if (samples <= 0) {
		return points;
	}

	points.reserve(samples);

	int side = static_cast<int>(std::ceil(std::sqrt(samples)));

	for (int i = 0; i < samples; ++i) {
		int row = i / side;
		int col = i % side;

		float x = (col + 0.5f) / side;
		float y = (row + 0.5f) / side;

		points.emplace_back(x, y);
	}

	return points;
}

std::vector<glm::vec2> generate_unifrom(int num_samples) {
	std::vector<glm::vec2> points;

	if (num_samples <= 0) {
		return points;
	}
	if (num_samples > 32) {
		spdlog::warn("too many samples generated, max size 32. Returning empty vector");
		return points;
	}
   
	points.reserve(num_samples);

    float numbers[64] = { 0.131538, 0.45865, 0.218959, 0.678865, 0.934693, 0.519416, 0.0345721, 0.5297, 0.00769819, 0.0668422, 0.686773, 0.930436, 0.526929, 0.653919, 0.701191, 0.762198, 0.0474645, 0.328234, 0.75641, 0.365339, 0.98255, 0.753356, 0.0726859, 0.884707, 0.436411, 0.477732, 0.274907, 0.166507, 0.897656, 0.0605643, 0.504523, 0.319033, 0.493977, 0.0907329, 0.0737491, 0.384142, 0.913817, 0.464446, 0.050084, 0.770205, 0.125365, 0.688455, 0.629543, 0.725412, 0.888572, 0.306322, 0.513274, 0.845982, 0.841511, 0.415395, 0.467917, 0.178328, 0.571655, 0.0330538, 0.49848, 0.748293, 0.890737, 0.84204, 0.212752, 0.130427, 0.274588, 0.414293, 0.70982, 0.239911 };

    /* generated from code in c++
#include <bits/stdc++.h>
using namespace std;
int main()
{
	// random generator
	default_random_engine gen;
	uniform_real_distribution<double> distribution(0.0,
												   1.0);


	cout << "{" ;
	for (int i = 0; i < 64; i++) {
		cout << distribution(gen) << ',';
	}

	cout << "}" ;

	return 0;
}
    */


    for (int i = 0; i < num_samples*2; i= i+2) {
		points.emplace_back(numbers[i], numbers[i + 1]);
	}
	return points;
}

void rotate_samples(std::vector<glm::vec2> &samples, float angle_rotation) {
	const float c = std::cos(angle_rotation);
	const float s = std::sin(angle_rotation);

	const glm::vec2 center(0.5f, 0.5f);

	for (glm::vec2 &p : samples) {
		glm::vec2 v = p - center;

		p.x = c * v.x - s * v.y + center.x;
		p.y = s * v.x + c * v.y + center.y;
	}
}

std::vector<glm::vec2> generate_circular_helix(int num_samples, float radius) {
	std::vector<glm::vec2> points;

	if (num_samples <= 0) {
		return points;
	}


	points.reserve(num_samples);

    radius = (radius < 0.5) ? radius :  0.5;

	float TWO_PI = 6.28318530718f;
	int half = (num_samples + 1) / 2;

	for (int i = 0; i < num_samples; i++) {
		int current_sample = (i % 2 == 0)
				? i / 2
				: half + i / 2;

		float t = TWO_PI * current_sample / num_samples;

		float x = 0.5f + radius * std::cos(t);
		float y = 0.5f + radius * std::sin(t);

		points.emplace_back(x, y);
	}

	return points;
}


std::vector<glm::vec2> generate_circular(int num_samples, float radius) {
	std::vector<glm::vec2> points;

	if (num_samples <= 0) {
		return points;
	}

	points.reserve(num_samples);

	radius = (radius < 0.5) ? radius : 0.5;

	float TWO_PI = 6.28318530718f;


	for (int i = 0; i < num_samples; i++) {


		float t = TWO_PI * i / num_samples;

		float x = 0.5f + radius * std::cos(t);
		float y = 0.5f + radius * std::sin(t);

		points.emplace_back(x, y);
	}

	return points;
}

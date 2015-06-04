#ifndef POINT_H
#define POINT_H

#pragma once

#include <vector>
using namespace std;

//Classe point
class Point
{
public:
	Point(int u, int v)
	{
		this->x = u;
		this->y = v;
	}

	Point (const Point &other)
	{
		this->x = other.x;
		this->y = other.y;
	}

	Point() {};

	int x;
	int y;
};

#endif

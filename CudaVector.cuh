#pragma once

#ifndef SAFE_DELETE
#define SAFE_DELETE(x) { if ((x)) { delete (x); x = nullptr; } }
#endif

#ifndef SAFE_DELETE_ARR
#define SAFE_DELETE_ARR(x) { if ((x)) { delete[] (x); x = nullptr; } }
#endif

// not thread safe
namespace Cuda {

	template <class T>
	class Vector
	{
	public:
		static const int CAPACITY_MIN = 32;

	public:
		__device__ Vector()
			: m_size(0)
			, m_capacity(0)
			, m_buffer(nullptr)
		{
		}

		__device__ Vector(const unsigned int capacity)
			: m_size(0)
			, m_capacity(0)
			, m_buffer(nullptr)
		{
			reserve(capacity);
			printf("\ncapacity : %d\n\n", m_capacity);
		}

		__device__ ~Vector()
		{
			SAFE_DELETE_ARR(m_buffer);
		}

	public:
		__device__ void reserve(const unsigned int& _capacity)
		{
			int capacity = _capacity;
			if (capacity < CAPACITY_MIN)
				capacity = CAPACITY_MIN;
			if (capacity <= m_capacity)
				return;

			unsigned int square = 1;
			while (square < capacity)
				square = square << 1;

			T* new_buffer = new T[sizeof(T) * square];
			for (unsigned int i = 0; i < m_size; i++)
				new_buffer[i] = m_buffer[i];

			SAFE_DELETE_ARR(m_buffer);

			m_buffer = new_buffer;
			m_capacity = square;
			printf("\nupdate capacity : %d\n\n", m_capacity);
		}

		__device__ void push_back(const T& val)
		{
			if (m_size >= m_capacity)
				reserve(m_size + 1);

			m_buffer[m_size++] = val;
		}

		__device__ void pop_back() 
		{
			if (empty())
				return;

			m_size--;
		}

		__device__ void clear()
		{
			SAFE_DELETE_ARR(m_buffer);
			m_capacity = 0;
			m_size = 0;
		}

		__device__ bool empty() const { return m_size == 0; }
		__device__ int size() { return m_size; }

		__device__ T& operator[](unsigned int idx) { return m_buffer[idx]; }
		__device__ T& front() { return m_buffer[0]; }
		__device__ T& back() { return m_buffer[m_size - 1]; }

	private:
		unsigned int m_log;
		unsigned int m_size;
		unsigned int m_capacity;
		T* m_buffer;
	};
}
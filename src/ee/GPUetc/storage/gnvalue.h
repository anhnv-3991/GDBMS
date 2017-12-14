#ifndef GNVALUE_H_
#define GNVALUE_H_

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "common/types.h"
#include <iostream>
#include <string>
#include <sstream>
#include "GPUetc/common/common.h"

namespace voltdb {

#define STRING_MASK 0x000000000000FFFF

class GNValue {
	friend class GTuple;
public:
	CUDAH GNValue();
	CUDAH GNValue(ValueType type);
	CUDAH GNValue(ValueType type, int64_t mdata, char *secondary_storage = NULL);
	CUDAH GNValue(ValueType type, const char *input, char *secondary_storage = NULL);

	CUDAD bool isNull() const;
	CUDAD bool isTrue() const;
	CUDAD bool isFalse() const;

	CUDAH void setValue(ValueType type, const char *data);
	CUDAH void setValueType(ValueType type);

	CUDAD int64_t getValue();
	CUDAD ValueType getValueType();

	CUDAD static GNValue getTrue();
	CUDAD static GNValue getFalse();
	CUDAD static GNValue getInvalid();
	CUDAD static GNValue getNullValue();


	CUDAD void setNull();

	CUDAD GNValue operator~() const;
	CUDAD GNValue operator&&(const GNValue rhs) const;
	CUDAD GNValue operator||(const GNValue rhs) const;
	CUDAD GNValue operator==(const GNValue rhs) const;
	CUDAD GNValue operator!=(const GNValue rhs) const;
	CUDAD GNValue operator<(const GNValue rhs) const;
	CUDAD GNValue operator<=(const GNValue rhs) const;
	CUDAD GNValue operator>(const GNValue rhs) const;
	CUDAD GNValue operator>=(const GNValue rhs) const;
	CUDAD GNValue operator+(const GNValue rhs) const;
	CUDAD GNValue operator*(const GNValue rhs) const;
	CUDAD GNValue operator-(const GNValue rhs) const;
	CUDAD GNValue operator/(const GNValue rhs) const;

	CUDAH int64_t getInt64() const;

	CUDAH double getDouble() const;
	CUDAH bool getBoolean() const;

	__forceinline__ std::string debug() const;
protected:
	int64_t m_data_;
	ValueType type_;
	char *secondary_storage_;
};

CUDAH GNValue::GNValue()
{
	m_data_ = 0;
	type_ = VALUE_TYPE_INVALID;
	secondary_storage_ = NULL;
}

CUDAH GNValue::GNValue(ValueType type)
{
	m_data_ = 0;
	type_ = type;
	secondary_storage_ = NULL;
}

CUDAH GNValue::GNValue(ValueType type, int64_t mdata, char *secondary_storage = NULL)
{
	m_data_ = mdata;
	type_ = type;
	secondary_storage_ = secondary_storage;
}

CUDAH GNValue::GNValue(ValueType type, const char *input, char *secondary_storage = NULL)
{
	type_ = type;

	switch (type) {
	case VALUE_TYPE_BOOLEAN:
	case VALUE_TYPE_TINYINT: {
		m_data_ = *reinterpret_cast<const int8_t *>(input);
		break;
	}
	case VALUE_TYPE_SMALLINT: {
		m_data_ = *reinterpret_cast<const int16_t *>(input);
		break;
	}
	case VALUE_TYPE_INTEGER: {
		m_data_ = *reinterpret_cast<const int32_t *>(input);
		break;
	}
	case VALUE_TYPE_BIGINT:
	case VALUE_TYPE_DOUBLE:
	case VALUE_TYPE_TIMESTAMP: {
		m_data_ = *reinterpret_cast<const int64_t *>(input);
		break;
	}
	default:
		m_data_ = 0;
		type_ = VALUE_TYPE_INVALID;
		break;
	}

	secondary_storage_ = secondary_storage;
}

CUDAD bool GNValue::isNull() const
{
	return (type_ == VALUE_TYPE_NULL);
}

CUDAD bool GNValue::isTrue() const
{
	return (type_ == VALUE_TYPE_BOOLEAN && (bool)m_data_);
}

CUDAD bool GNValue::isFalse() const
{
	return (type_ == VALUE_TYPE_BOOLEAN && !(bool)m_data_);
}

CUDAH void GNValue::setValue(ValueType type, const char *input)
{
	switch (type) {
	case VALUE_TYPE_BOOLEAN:
	case VALUE_TYPE_TINYINT: {
		m_data_ = *reinterpret_cast<const int8_t *>(input);
		break;
	}
	case VALUE_TYPE_SMALLINT: {
		m_data_ = *reinterpret_cast<const int16_t *>(input);
		break;
	}
	case VALUE_TYPE_INTEGER: {
		m_data_ = *reinterpret_cast<const int32_t *>(input);
		break;
	}
	case VALUE_TYPE_BIGINT:
	case VALUE_TYPE_DOUBLE:
	case VALUE_TYPE_TIMESTAMP: {
		m_data_ = *reinterpret_cast<const int64_t *>(input);
		break;
	}
	default: {
		break;
	}
	}
}

CUDAH void GNValue::setValueType(ValueType type)
{
	type_ = type;
}

CUDAD int64_t GNValue::getValue()
{
	return m_data_;
}

CUDAD ValueType GNValue::getValueType()
{
	return type_;
}


CUDAD GNValue GNValue::getTrue()
{
	bool value = true;
	return GNValue(VALUE_TYPE_BOOLEAN, (int64_t)value);
}

CUDAD GNValue GNValue::getFalse()
{
	bool value = false;
	return GNValue(VALUE_TYPE_BOOLEAN, (int64_t)value);
}

CUDAD GNValue GNValue::getInvalid()
{
	return GNValue(VALUE_TYPE_INVALID);
}

CUDAD GNValue GNValue::getNullValue()
{
	return GNValue(VALUE_TYPE_NULL);
}


CUDAD void GNValue::setNull()
{
	m_data_ = 0;
	type_ = VALUE_TYPE_NULL;
}


CUDAD GNValue GNValue::operator~() const
{
	if (type_ == VALUE_TYPE_BOOLEAN) {
		bool result = !((bool)m_data_);

		return GNValue(VALUE_TYPE_BOOLEAN, (int64_t)result);
	}

	return getInvalid();
}

CUDAD GNValue GNValue::operator&&(const GNValue rhs) const
{
	if (type_ == VALUE_TYPE_BOOLEAN && rhs.type_ == VALUE_TYPE_BOOLEAN) {
		bool result = (bool)m_data_ && (bool)(rhs.m_data_);

		return GNValue(VALUE_TYPE_BOOLEAN, (int64_t)result);
	}

	return getInvalid();
}

CUDAD GNValue GNValue::operator||(const GNValue rhs) const
{
	if (type_ == VALUE_TYPE_BOOLEAN && rhs.type_ == VALUE_TYPE_BOOLEAN) {
		bool result = (bool)m_data_ || (bool)(rhs.m_data_);

		return GNValue(VALUE_TYPE_BOOLEAN, (int64_t)result);
	}

	return getInvalid();
}

CUDAD GNValue GNValue::operator==(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		bool result;

		if (type_ == VALUE_TYPE_VARCHAR && rhs.type_ == VALUE_TYPE_VARCHAR) {
			int i = 0;
			result = true;
			int lsize = static_cast<int>(m_data_ & STRING_MASK);
			int rsize = static_cast<int>(rhs.m_data_ & STRING_MASK);
			char *left = secondary_storage_ + m_data_ >> 16;
			char *right = rhs.secondary_storage + rhs.m_data_ >> 16;

			while (i < lsize && i < rsize && result) {
				result = (left[i] == right[i]);
				i++;
			}
			__syncthreads();

			result = (lsize > i || rsize > i) ? false : result;
		} else if (type_ != VALUE_TYPE_VARCHAR && rhs.type_ != VALUE_TYPE_VARCHAR) {
			if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
				result = (getDouble() == rhs.getDouble());
			} else {
				result = (m_data_ == rhs.m_data_);
			}
		} else {
			return getInvalid();
		}

		return GNValue(VALUE_TYPE_BOOLEAN, (int64_t)result);
	}

	return getInvalid();
}

CUDAD GNValue GNValue::operator!=(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		bool result;

		if (type_ == VALUE_TYPE_VARCHAR && rhs.type_ == VALUE_TYPE_VARCHAR) {
			int i = 0;
			result = true;
			int lsize = static_cast<int>(m_data_ & STRING_MASK);
			int rsize = static_cast<int>(rhs.m_data_ & STRING_MASK);
			char *left = secondary_storage_ + m_data_ >> 16;
			char *right = rhs.secondary_storage + rhs.m_data_ >> 16;

			while (i < lsize && i < rsize && result) {
				result = (left[i] == right[i]);
				i++;
			}
			__syncthreads();

			result = (lsize == i && rsize == i) ? false : result;
		} else if (type_ != VALUE_TYPE_VARCHAR && rhs.type_ != VALUE_TYPE_VARCHAR) {
			if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
				result = (getDouble() != rhs.getDouble());
			} else {
				result = (m_data_ != rhs.m_data_);
			}
		} else {
			return getInvalid();
		}

		return GNValue(VALUE_TYPE_BOOLEAN, (int64_t)result);
	}

	return getInvalid();
}

CUDAD GNValue GNValue::operator<(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		bool result;

		if (type_ == VALUE_TYPE_VARCHAR && rhs.type_ == VALUE_TYPE_VARCHAR) {
			int i = 0;
			result = true;
			int lsize = static_cast<int>(m_data_ & STRING_MASK);
			int rsize = static_cast<int>(rhs.m_data_ & STRING_MASK);
			char *left = secondary_storage_ + m_data_ >> 16;
			char *right = rhs.secondary_storage + rhs.m_data_ >> 16;

			while (i < lsize && i < rsize && result) {
				result = (left[i] < right[i]);
				i++;
			}
			__syncthreads();

			result = (lsize > i && rsize > i) ? result : ((lsize >= i && rsize == i) ? false : true);

		} else if (type_ != VALUE_TYPE_VARCHAR && rhs.type != VALUE_TYPE_VARCHAR) {
			if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
				result = (getDouble() < rhs.getDouble());
			} else {
				result = (m_data_ < rhs.m_data_);
			}
		} else {
			return getInvalid();
		}

		return GNValue(VALUE_TYPE_BOOLEAN, (int64_t)result);
	}

	return getInvalid();
}

CUDAD GNValue GNValue::operator<=(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		bool result;

		if (type_ == VALUE_TYPE_VARCHAR && rhs.type_ == VALUE_TYPE_VARCHAR) {
			int i = 0;
			result = true;
			int lsize = static_cast<int>(m_data_ & STRING_MASK);
			int rsize = static_cast<int>(rhs.m_data_ & STRING_MASK);
			char *left = secondary_storage_ + m_data_ >> 16;
			char *right = rhs.secondary_storage + rhs.m_data_ >> 16;

			while (i < lsize && i < rsize && result) {
				result = (left[i] <= right[i]);
				i++;
			}
			__syncthreads();

			result = (lsize > i && rsize > i) ? result : ((lsize > i && rsize == i) ? false : true);

		} else if (type_ != VALUE_TYPE_VARCHAR && rhs.type != VALUE_TYPE_VARCHAR) {
			if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
				result = (getDouble() <= rhs.getDouble());
			} else {
				result = (m_data_ <= rhs.m_data_);
			}
		} else {
			return getInvalid();
		}

		return GNValue(VALUE_TYPE_BOOLEAN, (int64_t)result);
	}

	return getInvalid();
}

CUDAD GNValue GNValue::operator>(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		bool result;

		if (type_ == VALUE_TYPE_VARCHAR && rhs.type_ == VALUE_TYPE_VARCHAR) {
			int i = 0;
			result = true;
			int lsize = static_cast<int>(m_data_ & STRING_MASK);
			int rsize = static_cast<int>(rhs.m_data_ & STRING_MASK);
			char *left = secondary_storage_ + m_data_ >> 16;
			char *right = rhs.secondary_storage + rhs.m_data_ >> 16;

			while (i < lsize && i < rsize && result) {
				result = (left[i] > right[i]);
				i++;
			}
			__syncthreads();

			result = (lsize > i && rsize > i) ? result : ((lsize == i && rsize >= i) ? false : true);

		} else if (type_ != VALUE_TYPE_VARCHAR && rhs.type_ != VALUE_TYPE_VARCHAR) {
			if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
				result = (getDouble() > rhs.getDouble());
			} else {
				result = (m_data_ > rhs.m_data_);
			}
		} else {
			return getInvalid();
		}

		return GNValue(VALUE_TYPE_BOOLEAN, (int64_t)result);
	}

	return getInvalid();
}

CUDAD GNValue GNValue::operator>=(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		bool result;

		if (type_ == VALUE_TYPE_VARCHAR && rhs.type_ == VALUE_TYPE_VARCHAR) {
			int i = 0;
			result = true;
			int lsize = static_cast<int>(m_data_ & STRING_MASK);
			int rsize = static_cast<int>(rhs.m_data_ & STRING_MASK);
			char *left = secondary_storage_ + m_data_ >> 16;
			char *right = rhs.secondary_storage + rhs.m_data_ >> 16;

			while (i < lsize && i < rsize && result) {
				result = (left[i] >= right[i]);
				i++;
			}
			__syncthreads();

			result = (lsize > i && rsize > i) ? result : ((lsize == i && rsize > i) ? false : true);
		} else if (type_ != VALUE_TYPE_VARCHAR && rhs.type_ != VALUE_TYPE_VARCHAR) {
			if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
				result = (getDouble() >= rhs.getDouble());
			} else {
				result = (m_data_ >= rhs.m_data_);
			}
		} else {
			return getInvalid();
		}

		return GNValue(VALUE_TYPE_DOUBLE, (int64_t)result);
	}

	return getInvalid();
}

CUDAD GNValue GNValue::operator+(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		int64_t result;

		if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
			double res_d = getDouble() + rhs.getDouble();
			result = *reinterpret_cast<int64_t*>(&res_d);

			return GNValue(VALUE_TYPE_DOUBLE, result);
		} else {
			result = m_data_ + rhs.m_data_;
			ValueType res_type = (type_ > rhs.type_) ? type_ : rhs.type_;

			return GNValue(res_type, res_i);
		}
	}

	return getInvalid();
}

CUDAD GNValue GNValue::operator*(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		int64_t result;

		if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
			double res_d = getDouble() * rhs.getDouble();
			result = *reinterpret_cast<int64_t*>(&res_d);

			return GNValue(VALUE_TYPE_DOUBLE, result);
		} else {
			result = m_data_ * rhs.m_data_;
			ValueType res_type = (type_ > rhs.type_) ? type_ : rhs.type_;

			return GNValue(res_type, result);
		}
	}

	return getInvalid();
}

CUDAD GNValue GNValue::operator-(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		int64_t result;

		if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
			double res_d = getDouble() - rhs.getDouble();
			result = *reinterpret_cast<int64_t*>(&res_d);

			return GNValue(VALUE_TYPE_DOUBLE, result);
		} else {
			result = m_data_ - rhs.m_data_;
			ValueType res_type = (type_ > rhs.type_) ? type_ : rhs.type_;

			return GNValue(res_type, result);
		}
	}

	return getInvalid();
}

CUDAD GNValue GNValue::operator/(const GNValue rhs) const
{
	if (type_ != VALUE_TYPE_NULL && type_ != VALUE_TYPE_INVALID && rhs.type_ != VALUE_TYPE_NULL && rhs.type_ != VALUE_TYPE_INVALID) {
		int64_t result;
		ValueType res_type;

		if (type_ == VALUE_TYPE_DOUBLE || rhs.type_ == VALUE_TYPE_DOUBLE) {
			double left_d = getDouble();
			double right_d = rhs.getDouble();
			double res_d = (right_d != 0) ? left_d / right_d : DBL_MAX;

			result = *reinterpret_cast<int64_t*>(&res_d);
			res_type = (right_d != 0) ? VALUE_TYPE_DOUBLE : VALUE_TYPE_INVALID;
		} else {
			result = (rhs.m_data_ != 0) ? m_data_ / rhs.m_data_ : LLONG_MAX;
			res_type = (type_ > rhs.type_) ? type_ : rhs.type_;
			res_type = (rhs.m_data_ != 0) ? res_type : VALUE_TYPE_INVALID;
		}

		return GNValue(res_type, result);
	}

	return getInvalid();
}

__forceinline__ std::string GNValue::debug() const
{
	std::ostringstream output;

	switch (type_) {
	case VALUE_TYPE_INVALID: {
		output << "VALUE TYPE INVALID";
		break;
	}
	case VALUE_TYPE_NULL: {
		output << "VALUE TYPE NULL";
		break;
	}
	case VALUE_TYPE_FOR_DIAGNOSTICS_ONLY_NUMERIC: {
		output << "VALUE TYPE FOR DIAGNOSTICS ONLY NUMERIC";
		break;
	}
	case VALUE_TYPE_TINYINT: {
		output << "VALUE TYPE TINYINT: " << (int)m_data_;
		break;
	}
	case VALUE_TYPE_SMALLINT: {
		output << "VALUE TYPE SMALLINT: " << (int)m_data_;
		break;
	}
	case VALUE_TYPE_INTEGER: {
		output << "VALUE TYPE INTEGER: " << (int)m_data_;
		break;
	}
	case VALUE_TYPE_BIGINT: {
		output << "VALUE TYPE BIGINT: " << (int)m_data_;
		break;
	}
	case VALUE_TYPE_DOUBLE: {
		output << "VALUE TYPE DOUBLE: " << *reinterpret_cast<const double *>(&m_data_);
		break;
	}
	case VALUE_TYPE_VARCHAR: {
		output << "VALUE TYPE VARCHAR";
		break;
	}
	case VALUE_TYPE_TIMESTAMP: {
		output << "VALUE TYPE TIMESTAMP";
		break;
	}
	case VALUE_TYPE_DECIMAL: {
		output << "VALUE TYPE DECIMAL";
		break;
	}
	case VALUE_TYPE_BOOLEAN: {
		output << "VALUE TYPE BOOLEAN";
		break;
	}
	case VALUE_TYPE_ADDRESS: {
		output << "VALUE TYPE ADDRESS";
		break;
	}
	case VALUE_TYPE_VARBINARY: {
		output << "VALUE TYPE VARBINARY";
		break;
	}
	case VALUE_TYPE_ARRAY: {
		output << "VALUE TYPE VARBINARY";
		break;
	}
	default: {
		output << "UNDETECTED TYPE";
		break;
	}
	}

	std::string retval(output.str());

	return retval;
}

CUDAH int64_t GNValue::getInt64() const
{
	return m_data_;
}

CUDAH double GNValue::getDouble() const
{
	if (type_ == VALUE_TYPE_DOUBLE) {
		return *reinterpret_cast<double*>(&m_data_);
	} else {
		return static_cast<double>(m_data_);
	}
}

CUDAH bool GNValue::getBoolean() const
{
	if (type_ == VALUE_TYPE_BOOLEAN) {
		return static_cast<bool>(m_data_);
	} else {
		return false;
	}
}

}
#endif

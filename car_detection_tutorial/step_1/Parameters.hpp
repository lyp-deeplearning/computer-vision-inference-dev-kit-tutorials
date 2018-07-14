/*
 * Parameters.hpp
 *
 */

#ifndef PARAMETERS_HPP_
#define PARAMETERS_HPP_

#include <string.h>
#include <string>
#include <vector>
#include <stdint.h>

namespace parameters
{
/**
 * ParameterVal serves as base to generically process and access an argument
 */
class ParameterVal
{
public:
	ParameterVal(const char* name_str, const char* type_str, const char* desc_str)
         : name(name_str),
		   	   type(type_str),
			   desc(desc_str) {
   }

	virtual ~ParameterVal() {
	}

	virtual void printVal(std::ostream& os) {
		// do nothing, don't know size/type
	}

	virtual bool setVal(void* arg_val) {
		return false; // don't know size/type
	}

	virtual void* getVal() {
		// do nothing, don't know size/type
		return NULL;
	}

	virtual bool setVal(const char* arg_val) {
		return false; // don't know size/type
	}

   std::string name;
   std::string type;
   std::string desc;
};

/* List of all parameters, automatically populated by ParameterT constructor */
static std::vector<ParameterVal*> ParametersList;


/**
 * ParameterT tracks type and can handle accessing argument variable via underlying pointer
 */
template<typename T>
class ParameterT: public ParameterVal
{
public:
	ParameterT(const char* name_str, const char* type_str, const char* desc_str, T* t, T def_val)
         : ParameterVal(name_str, type_str, desc_str),
               val(t) {
      *val = def_val;
      ParametersList.push_back(this);
   }

   virtual ~ParameterT() {
   }

   void printVal(std::ostream& os) {
      os << *val;
   }

	virtual bool setVal(const char* arg_val) {
		T tmp;
		std::istringstream iss(arg_val);
		iss >> tmp;

		// if successful decode, set value, otherwise return fail
		return ((iss.rdstate() & std::ios::badbit) == 0) ? setVal(&tmp) : false;
	}

	virtual bool setVal(void* arg_val) {
		*val = *((T*) arg_val);
		return true;
	}

	virtual void* getVal() {
		return (void*) val;
	}

   T* val;
};

/**
* Find a parameter by name from the list
* @return pointer on success, NULL if not found
*/
static ParameterVal* findParameterByName(const char* param_name) {
	for (ParameterVal* paramT : ParametersList) {
		if (paramT->name.compare(param_name) == 0) {
			return paramT;
		}
	}
	return NULL;
}

/**
* Parses string of name=value pairs and sets variables found in the list
* Expected command line syntax is space separated pairs: <name>=<val>[ <name>=<val>...]
*/
static void parseParameterString(const char *p_str) {
    std::istringstream iss(p_str);
    std::string cfgKV;
    while (std::getline(iss, cfgKV, ' ')) {
    	// parse key and value pair
        size_t eqPos = cfgKV.find('=');
        if (std::string::npos == eqPos)
        {
           std::cout << "parseParameterString:ERROR finding '=' in key-value string, skipping string:"
        		   << cfgKV << std::endl;
           continue;
        }
        std::string keyStr = cfgKV.substr(0,eqPos);
        std::string valueStr(cfgKV.substr(eqPos+1));

        // find matching parameter name and set it
        ParameterVal* paramT = findParameterByName(keyStr.c_str());
		if (paramT != NULL) {
			// if value string begins with '$', treat it as a lookup of another parameter to use its value
			if ('$' == valueStr[0]) {
				ParameterVal* varParamT = findParameterByName(valueStr.c_str() + 1);
				if (varParamT != NULL) {
					if (paramT->type.compare(varParamT->type) == 0) {
						varParamT->printVal(std::cout);
						std::cout << std::endl;
						paramT->printVal(std::cout);
						std::cout << std::endl;

						paramT->setVal(varParamT->getVal());

						varParamT->printVal(std::cout);
						std::cout << std::endl;
						paramT->printVal(std::cout);
						std::cout << std::endl;
					} else {
				           std::cout << "parseParameterString:ERROR parameter name " << keyStr
				        		   << " and variable parameter name " << (valueStr.c_str() + 1)
								   << " are not the same type, skipping string:" << cfgKV << std::endl;
					}
				} else {
			           std::cout << "parseParameterString:ERROR finding variable parameter name " << (valueStr.c_str() + 1)
			        		   << " in key-value string, skipping string:" << cfgKV << std::endl;
				}
			} else {
				paramT->setVal(valueStr.c_str());
			}
		} else {
	           std::cout << "parseParameterString:ERROR finding parameter name " << keyStr <<
	        		   " in key-value string, skipping string:" << cfgKV << std::endl;
		}
    }
}

};
// namespace parameters

#define DEFINE_VARIABLE(type, name, value, help)             									\
  namespace parameters {                                             							\
    static type PARAMETERS_##name = value;                         								\
    static ParameterT<type> PARAMETERT_##name(#name, #type, help, &PARAMETERS_##name, value); 	\
  }                                                                     						\
  using parameters::PARAMETERS_##name;

/*
 * Used DEFINE_<type>(name, val, txt) macros to create parameters resulting in:
 * 	- Creating variable <type> PARAMETER_<name> to reference as parameter
 * 	- Variable's ParameterVal object may be found by <name> using:
 * 		static ParameterVal* findParameterByName(const char* param_name)
 * 	- Ability to set parameters by passing a null-terminated string of "<name>=<val>"
 * 		pairs separated by spaces ' ' to:
 * 			static void parseParameterString(const char *p_str)
 */
#define DEFINE_uint32(name, val, txt) \
   DEFINE_VARIABLE(uint32_t, name, val, txt)

#define DEFINE_int32(name, val, txt) \
	DEFINE_VARIABLE(int32_t, name, val, txt)

#define DEFINE_int64(name, val, txt) \
	DEFINE_VARIABLE(int64_t, name, val, txt)

#define DEFINE_uint64(name, val, txt) \
	DEFINE_VARIABLE(uint64_t, name, val, txt)

#define DEFINE_bool(name, val, txt) \
	DEFINE_VARIABLE(bool, name, val, txt)

#define DEFINE_double(name, val, txt) \
	DEFINE_VARIABLE(double, name, val, txt)

#define DEFINE_string(name, val, txt) \
	DEFINE_VARIABLE(std::string, name, val, txt)

#endif /* PARAMETERS_HPP_ */

#ifndef COMMON_OPTIONMANAGER_H
#define COMMON_OPTIONMANAGER_H

struct eqstrOptionManager
{
	bool operator()(const char* s1, const char* s2) const
	{
		return strcmp(s1, s2) == 0;
	}
};

struct greater_strOptionManager {
   bool operator()(const char* x, const char* y) const {
      if ( strcmp(x, y) < 0)
         return true;

      return false;
   }
};

#include <hash_map>
#include <iostream>

using namespace std;
using namespace stdext;

class OptionManager
{
public:
	
	/**
	 * Fetch option values for different option value types (all options are 
	 * string internally)
	 */
	const char *getOption(const char *group, const char *optionName, char *defaultValue = "");
	const int getOptionAsInt(const char *group, const char *optionName, const int defaultValue = 0);
	const float getOptionAsFloat(const char *group, const char *optionName, const float defaultValue = 0);
	const bool getOptionAsBool(const char *group, const char *optionName, const bool defaultValue = false);

	/**
	 * Set option values for different value types. Please note that all options
	 * will be saved as strings internally.
	 */
	bool setOption(const char *group, const char *optionName, const char *optionValue);
	bool setOptionAsInt(const char *group, const char *optionName, int optionValue);
	bool setOptionAsFloat(const char *group, const char *optionName, float optionValue);
	bool setOptionAsFloat(const char *group, const char *optionName, bool optionValue);
	
	/// Clear all option tables
	void clearOptions();

	/// Output all options (for debug purposes mostly)
	void dumpOptions();

	/// get Singleton instance or create it
	static OptionManager& getSingleton();
	/// get pointer to Singleton instance or create it
	static OptionManager* getSingletonPtr();
	
protected:
	/// Singleton, Ctor + Dtor protected
	OptionManager(const char* optionFile = "options.xml");
	~OptionManager();

	/// Singleton instance
	static OptionManager* m_Singleton;
	
	typedef stdext::hash_map<const char *, char *, stdext::hash_compare<const char*, greater_strOptionManager> > OptionList;
	typedef stdext::hash_map<const char *, OptionList *, stdext::hash_compare<const char*, greater_strOptionManager> > GroupList;

	/// Hash-Map containing the options and groups
	GroupList m_optionGroups;
	
private:
};



#endif
#include "tinyxml.h"
#include <iostream>
#include <stdlib.h>

#include "OptionManager.h"

using namespace std;

OptionManager* OptionManager::m_Singleton = 0;

OptionManager::OptionManager(const char* optionFile) {	

	// Open XML option file
	TiXmlDocument scene_doc( optionFile );
	if (scene_doc.LoadFile()) {
		TiXmlElement* pOptionItems = 0;

		TiXmlElement* pOptionRoot = scene_doc.RootElement();
		if (strcmp(pOptionRoot->Value(), "options") == 0)
			pOptionItems = pOptionRoot;
		else
			pOptionItems = pOptionRoot->FirstChildElement( "options" );
		
		// Error: no <options>-Tag found
		if (pOptionItems == 0) {
			cerr << "Warning: File '" << optionFile << "' does not seem to have any <options>-Tag inside !\nNo options will be set.\n";
			return;
		}

		// Fetch all <optiongroup>-Tags
		for( TiXmlNode* pOptionGroup = pOptionItems->FirstChild( "optiongroup" ); 
			 pOptionGroup != 0;
			 pOptionGroup = pOptionItems->IterateChildren("optiongroup", pOptionGroup) )
		{		
			// create new map for this group
			OptionList *newList = new OptionList();

			TiXmlElement* pOptionGroupElement = pOptionGroup->ToElement();
			const char *groupName = strdup(pOptionGroupElement->Attribute("name"));

			// Fetch all <option>-Tags for option group:
			for( TiXmlNode* pOption = pOptionGroup->FirstChild( "option" ); 
				 pOption != 0;
				 pOption = pOptionGroup->IterateChildren("option", pOption) )
			{
				TiXmlElement* pOptionElement = pOption->ToElement();

				const char *optionName = strdup(pOptionElement->Attribute("name"));   // Name
				char *optionValue = strdup(pOptionElement->Attribute("value")); // Value
				
				if (optionName == 0)
					continue;

				// insert option into option list
				(*newList)[optionName] = optionValue;
			}

			// insert option map into group list
			m_optionGroups[groupName] = newList;
		}
	}
	else {
		cerr << "Warning: Could not read option file '" << optionFile << "' (not found or invalid XML syntax).\nNo options will be set.\n";
		cout << scene_doc.ErrorDesc();		
	}

}

OptionManager::~OptionManager() {
	clearOptions();
}

OptionManager* OptionManager::getSingletonPtr()
{
	if (m_Singleton == 0)
		m_Singleton = new OptionManager();
	return m_Singleton;
}

OptionManager& OptionManager::getSingleton()
{  
	if (m_Singleton == 0)
		m_Singleton = new OptionManager();
	return *m_Singleton;
}

/**
* Fetch option values for different option value types (all options are 
* string internally)
*/
const char *OptionManager::getOption(const char *group, const char *optionName, char *defaultValue) {
	OptionList::iterator optionIt;
	GroupList::iterator groupIt;

	groupIt = m_optionGroups.find(group);
	if (groupIt == m_optionGroups.end()) // option group not found		
		return defaultValue;

	optionIt = groupIt->second->find(optionName);
	if (optionIt == groupIt->second->end()) // option not found
		return defaultValue;
	else
		return optionIt->second;
}

const int OptionManager::getOptionAsInt(const char *group, const char *optionName, const int defaultValue) {
	const char *res = getOption(group, optionName, NULL);

	if (res == NULL) // not found
		return defaultValue;
	else
		return atoi(res);
}

const float OptionManager::getOptionAsFloat(const char *group, const char *optionName, const float defaultValue) {
	const char *res = getOption(group, optionName, NULL);

	if (res == NULL) // not found
		return defaultValue;
	else
		return (float)atof(res);	
}

const bool OptionManager::getOptionAsBool(const char *group, const char *optionName, const bool defaultValue) {
	const char *res = getOption(group, optionName, NULL);

	if (res == NULL) // not found
		return defaultValue;
	else
		return (atoi(res) != 0);	
}

/**
* Set option values for different value types. Please note that all options
* will be saved as strings internally.
*/
bool OptionManager::setOption(const char *group, const char *optionName, const char *optionValue) {
	return true;
}
bool OptionManager::setOptionAsInt(const char *group, const char *optionName, int optionValue) {
	return true;
}
bool OptionManager::setOptionAsFloat(const char *group, const char *optionName, float optionValue) {
	return true;
}
bool OptionManager::setOptionAsFloat(const char *group, const char *optionName, bool optionValue) {
	return true;
}

/// Clear all option tables
void OptionManager::clearOptions() {
	// delete all created option groups
	GroupList::iterator it;
	OptionList::iterator it2;
	for (it = m_optionGroups.begin(); it != m_optionGroups.end(); ++it)
	{		
		it->second->clear();
	}
	m_optionGroups.clear();
}

/// Output all options (for debug purposes mostly)
void OptionManager::dumpOptions() {
	GroupList::iterator it;
	OptionList::iterator it2;

	// for all option groups
	for (it = m_optionGroups.begin(); it != m_optionGroups.end(); ++it)
	{		
		cout << "[" << it->first << "]\n";
		
		// for all options
		for (it2 = it->second->begin(); it2 != it->second->end(); ++it2)
		{
			cout << " > " << it2->first << "\t=\t'" << it2->second << "'\n";
		}
	}
}
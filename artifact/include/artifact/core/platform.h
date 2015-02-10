#ifndef PLATFORM_H_
#define PLATFORM_H_

#include "config.h"
#include <string>
#include <log4cpp/Category.hh>
#undef int64_t

#define LOKI_OBJECT_LEVEL_THREADING

#include <loki/Singleton.h>

#include <unordered_map>

#if defined USE_MATLAB
#include <engine.h>
#endif

#include <boost/filesystem.hpp>

namespace core
{



	using namespace std;

	class config_item
	{
	};

	enum LogType
	{
	  Running,
	  Debug,
	  Output,
	  Temp
	};

	class POCO_EXPORT platform_
	{
	private:

#if defined USE_MATLAB
		Engine *ep;
#endif

		unordered_map<string,shared_ptr<config_item> > configs;
		unordered_map<string,log4cpp::Category* > logs;
		//int train_valid_pair_id;
		//int train_test_pair_id;

		string experimentID;

		boost::filesystem::path settingDirPath;

		boost::filesystem::path tempDirPath;

		boost::filesystem::path resultDirPath;

	public:

		platform_();
		~platform_(void);

		void init();

		void shutdown();

		/* log4cpp::Category& get_output_log(); */
		/* log4cpp::Category& get_running_log(); */
		/* log4cpp::Category& get_debug_log(); */

		log4cpp::Category& get_log(const LogType & logType, const string & logName);
#if defined USE_MATLAB
		Engine * get_matlab_eigen();
#endif

		shared_ptr<config_item> get_config(const string & key);

		void set_config(const string & key, const shared_ptr<config_item> & config);

		void setExperimentID(const string& experimentID);
		const string & getExperimentID() const;

		void setSettingDirPath(const boost::filesystem::path& path);
		const boost::filesystem::path & getSettingDirPath() const;

		void setTempDirPath(const boost::filesystem::path& path);
		const boost::filesystem::path & getTempDirPath() const;

		void setResultDirPath(const boost::filesystem::path& path);
		const boost::filesystem::path & getResultDirPath() const;

		boost::filesystem::path getCurrentSettingDirPath() const;
		boost::filesystem::path getCurrentResultDirPath() const;
		boost::filesystem::path getCurrentTempDirPath() const;

	};

	typedef Loki::SingletonHolder<platform_,
	  Loki::CreateUsingNew,
	  Loki::DefaultLifetime,
	  Loki::ClassLevelLockable
	> platform;

}



#endif

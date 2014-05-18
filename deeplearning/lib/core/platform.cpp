#include <liblearning/core/platform.h>
#include <liblearning/util/math_util.h>
#include <liblearning/core/serialize.h>
#include <liblearning/core/dataset.h>
#include <liblearning/core/supervised_dataset.h>
#include <liblearning/experiment/train_validation_pair.h>
#include <liblearning/experiment/train_test_pair.h>
#include <liblearning/experiment/experiment_datasets.h>
#include <liblearning/deep/deep_auto_encoder.h>

#include <log4cpp/FileAppender.hh>
#include <log4cpp/PatternLayout.hh>
#include <log4cpp/SimpleLayout.hh>

#ifdef USE_GPU
#include <cublas.h>
#include <cuda.h>
 
#endif
using namespace core;

using namespace experiment;
using namespace deep;
using namespace boost::filesystem;

platform_::platform_()
{
#if defined USE_MATLAB
	ep = 0;
#endif	
}


platform_::~platform_(void)
{

}

void platform_::init()
{


	init_math_utils();
	
	camp::Class::declare<dataset>("dataset").constructor0();
	camp::Class::declare<supervised_dataset>("supervised_dataset").constructor0().base<dataset>();

	camp::Class::declare<train_validation_pair>("train_validation_pair").constructor0();
	camp::Class::declare<train_test_pair>("train_test_pair").constructor0();
	camp::Class::declare<experiment_datasets>("experiment_datasets").constructor0();
	camp::Class::declare<deep_auto_encoder>("deep_auto_encoder").constructor0();

#ifdef USE_GPU
	
	cublasStatus status = cublasInit();

	if (status != CUBLAS_STATUS_SUCCESS)
		throw runtime_error("cublas initialize failed");

#endif 

	// log4cpp::Appender *appender_out = new log4cpp::FileAppender("FileAppender1",running_name + ".Out");
	// log4cpp::Appender *appender_run = new log4cpp::FileAppender("FileAppender2",running_name + ".Run");
	// log4cpp::Appender *appender_debug = new log4cpp::FileAppender("FileAppender3",running_name + ".Debug");
 
	// log4cpp::Layout *layout_out = new log4cpp::SimpleLayout();
	// log4cpp::Layout *layout_run = new log4cpp::BasicLayout();
	// log4cpp::Layout *layout_debug = new log4cpp::BasicLayout();

	// log4cpp::Category& category_out = log4cpp::Category::getInstance("Category_Out");
	// log4cpp::Category& category_run = log4cpp::Category::getInstance("Category_Run");
	// log4cpp::Category& category_debug = log4cpp::Category::getInstance("Category_Debug");

	// category_out.setPriority(log4cpp::Priority::INFO);
	// category_run.setPriority(log4cpp::Priority::INFO);
	// category_debug.setPriority(log4cpp::Priority::DEBUG);

	// appender_out->setLayout(layout_out);
	// appender_run->setLayout(layout_run);
	// appender_debug->setLayout(layout_debug);

	// category_out.addAppender(appender_out);
	// category_run.addAppender(appender_run);
	// category_debug.addAppender(appender_debug);
#if defined USE_MATLAB

	if (!(ep = engOpen("\0"))) {
		ep = 0;
	}

#endif	

}
#if defined USE_MATLAB
Engine * platform_::get_matlab_eigen()
{
	return ep;
}
#endif

void platform_::shutdown()
{
	finish_math_utils();
#ifdef USE_GPU
	cublasStatus status = cublasShutdown();

	if (status != CUBLAS_STATUS_SUCCESS)
		throw runtime_error("cublas shutdown failed");
#endif 

#if defined USE_MATLAB
	if (ep != 0)
		engClose(ep);
#endif
}


shared_ptr<config_item> platform_::get_config(const string & key)
{
  unordered_map<string,shared_ptr<config_item>>::iterator result = configs.find(key);
  
  if (result == configs.end())
  {
    throw runtime_error("The configuation named as " + key + " not found!");
  }
  
  return (* result).second;
}
		
		
void platform_::set_config(const string & key, const shared_ptr<config_item> & config)
{
  configs[key] = config;
}

		void  platform_::setExperimentID(const string& experimentID_)
		{
		  experimentID = experimentID_;
		}

		const string &  platform_::getExperimentID() const
		{
		  return experimentID;		  
		}

		void  platform_::setSettingDirPath(const path& path)
		{
		  settingDirPath = path;

		  if (!exists(settingDirPath))
		    create_directories(settingDirPath);

		}
		const path &  platform_::getSettingDirPath() const
		{
		  return settingDirPath;
		}

		void  platform_::setTempDirPath(const path& path)
		{
		  tempDirPath = path;
		  if (!exists(tempDirPath))
		    create_directories(tempDirPath);
		}
		const path &  platform_::getTempDirPath() const
		{
		  return tempDirPath;
		}

		void  platform_::setResultDirPath(const path& path)
		{
		  resultDirPath = path;
		  if (!exists(resultDirPath))
		    create_directories(resultDirPath);
		}
		const path &  platform_::getResultDirPath() const
		{
		  return resultDirPath;
		}

		path platform_::getCurrentSettingDirPath() const
		{
		  auto currentSettingDirPath = settingDirPath / path(experimentID);
		  if (!exists(currentSettingDirPath))
		    create_directories(currentSettingDirPath);
		  return currentSettingDirPath;
		}
		path platform_::getCurrentResultDirPath() const
		{

		  auto currentResultDirPath = resultDirPath / path(experimentID);
		  if (!exists(currentResultDirPath))
		    create_directories(currentResultDirPath);
		  return currentResultDirPath;
		}
		path platform_::getCurrentTempDirPath() const
		{
		  auto currentTempDirPath = tempDirPath / path(experimentID);
		  if (!exists(currentTempDirPath))
		    create_directories(currentTempDirPath);
		  return currentTempDirPath;
		}

// log4cpp::Category& platform_::get_output_log()
// {
// 	return log4cpp::Category::getInstance("Category_Out");
// }
// log4cpp::Category& platform_::get_running_log()
// {
// 	return log4cpp::Category::getInstance("Category_Run");
// }
// log4cpp::Category& platform_::get_debug_log()
// {
// 	return log4cpp::Category::getInstance("Category_Debug");
// }


log4cpp::Category& platform_::get_log(const LogType & logType, const string & logName)
{
  path logPath;
  
  log4cpp::Priority::PriorityLevel priority;


  if (logType == Running)
    {
      logPath = ".";
      priority = log4cpp::Priority::INFO;
    }
  else if (logType == Debug)
    {
      logPath = ".";
      priority = log4cpp::Priority::DEBUG;
    }
  else if (logType == Output)
    {
      logPath = getCurrentResultDirPath();
      priority = log4cpp::Priority::INFO;
    }
  else if (logType == Temp)
    {
      logPath = getCurrentTempDirPath();
      priority = log4cpp::Priority::INFO;
    }

  logPath /= path(logName + ".log");

  string logID = logPath.string();
	
//	cout << "log:" << logID << endl;

  if (logs.find(logID) == logs.end())

    {

      log4cpp::Appender *appender = new log4cpp::FileAppender("FileAppender1",logPath.string());

		log4cpp::PatternLayout* layout = new log4cpp::PatternLayout();
		layout->setConversionPattern("%d: %p - %m %n");
      //layout->setConversionPattern("%d: %p %c %x: %m%n");

      log4cpp::Category& category = log4cpp::Category::getInstance(logPath.string());

      category.setPriority(priority);

      appender->setLayout(layout);

      category.addAppender(appender);

      logs[logID] = &category;
    }


    return *logs[logID];

}

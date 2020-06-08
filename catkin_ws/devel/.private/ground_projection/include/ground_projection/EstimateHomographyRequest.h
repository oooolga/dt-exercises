// Generated by gencpp from file ground_projection/EstimateHomographyRequest.msg
// DO NOT EDIT!


#ifndef GROUND_PROJECTION_MESSAGE_ESTIMATEHOMOGRAPHYREQUEST_H
#define GROUND_PROJECTION_MESSAGE_ESTIMATEHOMOGRAPHYREQUEST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace ground_projection
{
template <class ContainerAllocator>
struct EstimateHomographyRequest_
{
  typedef EstimateHomographyRequest_<ContainerAllocator> Type;

  EstimateHomographyRequest_()
    {
    }
  EstimateHomographyRequest_(const ContainerAllocator& _alloc)
    {
  (void)_alloc;
    }







  typedef boost::shared_ptr< ::ground_projection::EstimateHomographyRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::ground_projection::EstimateHomographyRequest_<ContainerAllocator> const> ConstPtr;

}; // struct EstimateHomographyRequest_

typedef ::ground_projection::EstimateHomographyRequest_<std::allocator<void> > EstimateHomographyRequest;

typedef boost::shared_ptr< ::ground_projection::EstimateHomographyRequest > EstimateHomographyRequestPtr;
typedef boost::shared_ptr< ::ground_projection::EstimateHomographyRequest const> EstimateHomographyRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::ground_projection::EstimateHomographyRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::ground_projection::EstimateHomographyRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}


} // namespace ground_projection

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::ground_projection::EstimateHomographyRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::ground_projection::EstimateHomographyRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::ground_projection::EstimateHomographyRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::ground_projection::EstimateHomographyRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::ground_projection::EstimateHomographyRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::ground_projection::EstimateHomographyRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::ground_projection::EstimateHomographyRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "d41d8cd98f00b204e9800998ecf8427e";
  }

  static const char* value(const ::ground_projection::EstimateHomographyRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xd41d8cd98f00b204ULL;
  static const uint64_t static_value2 = 0xe9800998ecf8427eULL;
};

template<class ContainerAllocator>
struct DataType< ::ground_projection::EstimateHomographyRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "ground_projection/EstimateHomographyRequest";
  }

  static const char* value(const ::ground_projection::EstimateHomographyRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::ground_projection::EstimateHomographyRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "\n"
;
  }

  static const char* value(const ::ground_projection::EstimateHomographyRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::ground_projection::EstimateHomographyRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream&, T)
    {}

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct EstimateHomographyRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::ground_projection::EstimateHomographyRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream&, const std::string&, const ::ground_projection::EstimateHomographyRequest_<ContainerAllocator>&)
  {}
};

} // namespace message_operations
} // namespace ros

#endif // GROUND_PROJECTION_MESSAGE_ESTIMATEHOMOGRAPHYREQUEST_H
#pragma once

#include "tf_base.h"

namespace tensorflow_c_api
{
    class operation : private not_copyable
    {

    public:

        operation() : m_value(nullptr)
        {

        }

        operation(TF_Operation* b) : m_value(b)
        {
        }

        ~operation()
        {

        }

        operation(operation&& o)
        {
            m_value = o.m_value;
            o.m_value = nullptr;
        }

        operation& operator=(operation&& o)
        {
            if (&o != this)
            {
                m_value = o.m_value;
                o.m_value = nullptr;
            }
        }

        operator TF_Operation *() const
        {
            return m_value;
        }

        const char* name() const
        {
            return TF_OperationName(m_value);
        }

        const char* type() const
        {
            return TF_OperationOpType(m_value);
        }

        const char* device() const
        {
            return TF_OperationDevice(m_value);
        }

        int32_t num_outputs() const
        {
            return TF_OperationNumOutputs(m_value);
        }

        int32_t num_inputs() const
        {
            return TF_OperationNumInputs(m_value);
        }

        /*
        TF_CAPI_EXPORT extern const char* TF_OperationName(TF_Operation* oper);
        TF_CAPI_EXPORT extern const char* TF_OperationOpType(TF_Operation* oper);
        TF_CAPI_EXPORT extern const char* TF_OperationDevice(TF_Operation* oper);

        TF_CAPI_EXPORT extern int TF_OperationNumOutputs(TF_Operation* oper);
        TF_CAPI_EXPORT extern TF_DataType TF_OperationOutputType(TF_Output oper_out);
        TF_CAPI_EXPORT extern int TF_OperationOutputListLength(TF_Operation* oper,
            const char* arg_name,
            TF_Status* status);

        TF_CAPI_EXPORT extern int TF_OperationNumInputs(TF_Operation* oper);
        TF_CAPI_EXPORT extern TF_DataType TF_OperationInputType(TF_Input oper_in);
        TF_CAPI_EXPORT extern int TF_OperationInputListLength(TF_Operation* oper,
            const char* arg_name,
            TF_Status* status);

        // In this code:
        //   TF_Output producer = TF_OperationInput(consumer);
        // There is an edge from producer.oper's output (given by
        // producer.index) to consumer.oper's input (given by consumer.index).
        TF_CAPI_EXPORT extern TF_Output TF_OperationInput(TF_Input oper_in);

        // Get the number of current consumers of a specific output of an
        // operation.  Note that this number can change when new operations
        // are added to the graph.
        TF_CAPI_EXPORT extern int TF_OperationOutputNumConsumers(TF_Output oper_out);

        // Get list of all current consumers of a specific output of an
        // operation.  `consumers` must point to an array of length at least
        // `max_consumers` (ideally set to
        // TF_OperationOutputNumConsumers(oper_out)).  Beware that a concurrent
        // modification of the graph can increase the number of consumers of
        // an operation.  Returns the number of output consumers (should match
        // TF_OperationOutputNumConsumers(oper_out)).
        TF_CAPI_EXPORT extern int TF_OperationOutputConsumers(TF_Output oper_out,
            TF_Input* consumers,
            int max_consumers);

        // Get the number of control inputs to an operation.
        TF_CAPI_EXPORT extern int TF_OperationNumControlInputs(TF_Operation* oper);

        // Get list of all control inputs to an operation.  `control_inputs` must
        // point to an array of length `max_control_inputs` (ideally set to
        // TF_OperationNumControlInputs(oper)).  Returns the number of control
        // inputs (should match TF_OperationNumControlInputs(oper)).
        TF_CAPI_EXPORT extern int TF_OperationGetControlInputs(
            TF_Operation* oper, TF_Operation** control_inputs, int max_control_inputs);

        // Get the number of operations that have `*oper` as a control input.
        // Note that this number can change when new operations are added to
        // the graph.
        TF_CAPI_EXPORT extern int TF_OperationNumControlOutputs(TF_Operation* oper);

        // Get the list of operations that have `*oper` as a control input.
        // `control_outputs` must point to an array of length at least
        // `max_control_outputs` (ideally set to
        // TF_OperationNumControlOutputs(oper)). Beware that a concurrent
        // modification of the graph can increase the number of control
        // outputs.  Returns the number of control outputs (should match
        // TF_OperationNumControlOutputs(oper)).
        TF_CAPI_EXPORT extern int TF_OperationGetControlOutputs(
            TF_Operation* oper, TF_Operation** control_outputs,
            int max_control_outputs);
        */

    private:

        TF_Operation* m_value;
    };

    struct output
    {
        operation       m_o;
        ::std::int32_t  m_index;

        operator TF_Output () const
        {
            return TF_Output{ m_o, m_index };
        }
    };

    struct input
    {
        operation       m_o;
        ::std::int32_t  m_index;

        operator TF_Output () const
        {
            return TF_Output{ m_o, m_index };
        }
    };
}
